import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from schemas import SimilarityMethod, BatchUploadResponse, BatchStatus


class BatchDocumentClient:
    """Client for batch document processing and similarity analysis."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with the API base URL."""
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/v1"
    
    def upload_batch(
        self,
        files: List[str],
        method: SimilarityMethod = SimilarityMethod.TFIDF,
        threshold: float = 0.8,
        model_name: Optional[str] = None
    ) -> str:
        """
        Upload a batch of documents for processing.
        
        Args:
            files: List of file paths to upload
            method: Similarity method to use
            threshold: Similarity threshold for considering documents as duplicates
            model_name: Optional name of the sentence transformer model to use
            
        Returns:
            Batch ID for tracking the processing status
        """
        # Prepare files for upload
        file_objs = []
        for file_path in files:
            with open(file_path, 'rb') as f:
                file_objs.append(('files', (Path(file_path).name, f, 'application/octet-stream')))
        
        # Upload files
        response = requests.post(
            f"{self.api_url}/compare_selected",
            files=file_objs,
            data={
                'method': method,
                'threshold': threshold,
                'model_name': model_name
            }
        )
        response.raise_for_status()
        
        batch_response = BatchUploadResponse(**response.json())
        return batch_response.batch_id
    
    def compare_all_files(
        self,
        store_path: str,
        method: SimilarityMethod = SimilarityMethod.TFIDF,
        threshold: float = 0.8,
        model_name: Optional[str] = None
    ) -> str:
        """
        Compare all files in a store directory.
        
        Args:
            store_path: Path to the directory containing files to compare
            method: Similarity method to use
            threshold: Similarity threshold for considering documents as duplicates
            model_name: Optional name of the sentence transformer model to use
            
        Returns:
            Batch ID for tracking the processing status
        """
        response = requests.post(
            f"{self.api_url}/compare_all",
            data={
                'store_path': store_path,
                'method': method,
                'threshold': threshold,
                'model_name': model_name
            }
        )
        response.raise_for_status()
        
        batch_response = BatchUploadResponse(**response.json())
        return batch_response.batch_id
    
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get the current status of a batch processing job."""
        response = requests.get(f"{self.api_url}/batch_status/{batch_id}")
        response.raise_for_status()
        return BatchStatus(**response.json())
    
    def wait_for_completion(self, batch_id: str, poll_interval: int = 5) -> BatchStatus:
        """
        Wait for batch processing to complete.
        
        Args:
            batch_id: ID of the batch to monitor
            poll_interval: Time between status checks in seconds
            
        Returns:
            Final status of the batch processing
        """
        while True:
            status = self.get_batch_status(batch_id)
            if status.status in ['completed', 'error']:
                return status
            time.sleep(poll_interval) 