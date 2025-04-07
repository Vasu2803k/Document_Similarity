from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import logging
import json
import uuid
import os
import aiofiles
import asyncio
from models.document_processor import DocumentProcessor
from models.batch_analyzer import BatchSimilarityAnalyzer
from models.vector_store import DocumentVectorStore
from models.schemas import SimilarityMethod, BatchStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize components
vector_store = DocumentVectorStore()
document_processor = DocumentProcessor(upload_dir=UPLOAD_DIR)
batch_analyzer = BatchSimilarityAnalyzer(
    method="embedding",
    threshold=0.8,
    vector_store=vector_store,
    document_processor=document_processor
)

app = FastAPI(title="Document Similarity API")

async def save_uploaded_file(file: UploadFile, file_path: Path) -> None:
    """Asynchronously save an uploaded file."""
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

async def save_results(batch_id: str, result: dict) -> None:
    """Asynchronously save processing results."""
    result_file = RESULTS_DIR / f"{batch_id}.json"
    async with aiofiles.open(result_file, 'w') as f:
        await f.write(json.dumps(result))

async def cleanup_files(file_paths: List[str]) -> None:
    """Asynchronously clean up temporary files."""
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Error cleaning up file {file_path}: {str(e)}")

@app.post("/api/compare")
async def compare_documents(
    text1: str,
    text2: str,
    method: SimilarityMethod = SimilarityMethod.embedding,
    threshold: float = 0.8
):
    """Compare two text documents."""
    try:
        similarity = await batch_analyzer.compare_texts(text1, text2, method.value, threshold)
        return {"similarity": similarity}
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-compare")
async def upload_compare(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    method: SimilarityMethod = SimilarityMethod.embedding,
    threshold: float = 0.8
):
    """Upload and compare two files."""
    try:
        # Save files
        file1_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file1.filename}"
        file2_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file2.filename}"
        
        await asyncio.gather(
            save_uploaded_file(file1, file1_path),
            save_uploaded_file(file2, file2_path)
        )
        
        # Process and compare
        similarity = await batch_analyzer.compare_files(
            str(file1_path),
            str(file2_path),
            method.value,
            threshold
        )
        
        # Cleanup
        await cleanup_files([str(file1_path), str(file2_path)])
        
        return {"similarity": similarity}
    except Exception as e:
        logger.error(f"Error comparing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    method: SimilarityMethod = SimilarityMethod.embedding,
    threshold: float = 0.8,
    model_name: Optional[str] = None
):
    """Upload multiple files for batch processing."""
    try:
        # Save files
        file_paths = []
        save_tasks = []
        
        for file in files:
            file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
            file_paths.append(str(file_path))
            save_tasks.append(save_uploaded_file(file, file_path))
        
        await asyncio.gather(*save_tasks)
        
        # Process files
        result = await batch_analyzer.analyze_selected_files(
            file_paths,
            method.value,
            threshold,
            model_name
        )
        
        # Save results
        batch_id = str(uuid.uuid4())
        await save_results(batch_id, result)
        
        # Cleanup
        await cleanup_files(file_paths)
        
        return {"batch_id": batch_id, "status": "completed"}
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare-selected")
async def compare_selected_files(
    file_paths: List[str],
    method: SimilarityMethod = SimilarityMethod.embedding,
    threshold: float = 0.8,
    model_name: Optional[str] = None
):
    """Compare selected files."""
    try:
        result = await batch_analyzer.analyze_selected_files(
            file_paths,
            method.value,
            threshold,
            model_name
        )
        
        # Save results
        batch_id = str(uuid.uuid4())
        await save_results(batch_id, result)
        
        return {"batch_id": batch_id, "status": "completed"}
    except Exception as e:
        logger.error(f"Error comparing selected files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """Get the status of a batch processing job."""
    try:
        result_file = RESULTS_DIR / f"{batch_id}.json"
        if not result_file.exists():
            return {"status": "not_found"}
        
        async with aiofiles.open(result_file, 'r') as f:
            result = json.loads(await f.read())
        
        return {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    ) 