# Document Similarity Analysis

A powerful tool for analyzing document similarity using FAISS vector store and LangChain for efficient document processing and similarity search.

## Features

- **Document Processing**
  - Support for multiple document types (PDF, DOCX, TXT)
  - Table extraction from PDFs and DOCX files
  - Configurable text cleaning and preprocessing
  - Robust error handling and logging

- **Similarity Analysis**
  - Multiple similarity methods:
    - TF-IDF based similarity
    - Embedding-based similarity using sentence transformers
    - Sophisticated embedding models for advanced comparison
  - Configurable similarity thresholds
  - Batch processing of multiple documents
  - Detailed similarity metrics and summaries

- **Vector Storage**
  - Efficient document storage using FAISS
  - Document chunking with configurable parameters
  - Metadata management for documents
  - Automatic cleanup of old documents

- **API Interface**
  - RESTful API endpoints for document processing and analysis
  - Support for file uploads and batch processing
  - Background task processing for large datasets
  - Detailed status tracking and error reporting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-similarity.git
cd document-similarity
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
uvicorn api.main:app --reload
```

### API Endpoints

1. **Compare Documents**
```bash
POST /api/compare
Content-Type: application/json

{
    "text1": "First document text",
    "text2": "Second document text",
    "method": "embedding",
    "threshold": 0.8
}
```

2. **Upload and Compare Files**
```bash
POST /api/upload-compare
Content-Type: multipart/form-data

file1: <file>
file2: <file>
method: embedding
threshold: 0.8
```

3. **Batch Upload**
```bash
POST /api/upload-batch
Content-Type: multipart/form-data

files: <multiple files>
method: embedding
threshold: 0.8
model_name: optional_model_name
```

4. **Compare Selected Files**
```bash
POST /api/compare-selected
Content-Type: application/json

{
    "file_paths": ["path/to/file1", "path/to/file2"],
    "method": "embedding",
    "threshold": 0.8
}
```

5. **Get Batch Status**
```bash
GET /api/batch/{batch_id}/status
```

### Python Client Usage

```python
from models.document_processor import DocumentProcessor
from models.batch_analyzer import BatchSimilarityAnalyzer
from models.vector_store import DocumentVectorStore

# Initialize components
vector_store = DocumentVectorStore()
document_processor = DocumentProcessor(
    upload_dir="uploads",
    skip_images=True,
    extract_tables=True
)
batch_analyzer = BatchSimilarityAnalyzer(
    method="embedding",
    threshold=0.8,
    vector_store=vector_store,
    document_processor=document_processor
)

# Process and analyze documents
result = batch_analyzer.analyze_directory(
    "path/to/documents",
    output_file="results.csv"
)

# Get similarity summary
summary = batch_analyzer.get_similarity_summary(result)
```

## Configuration

### Document Processor
```python
DocumentProcessor(
    upload_dir="uploads",      # Directory for uploaded files
    skip_images=True,          # Skip image processing
    extract_tables=True,       # Extract tables from documents
    logging_level=logging.INFO # Logging level
)
```

### Vector Store
```python
DocumentVectorStore(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
    chunk_size=1000,           # Size of text chunks
    chunk_overlap=200,         # Overlap between chunks
    store_path="vector_store", # Path to store vectors
    dimension=384,             # Dimension of embedding vectors
    similarity_metric="cosine",# Similarity metric
    max_documents=1000000,     # Maximum documents to store
    logging_level=logging.INFO # Logging level
)
```

### Batch Analyzer
```python
BatchSimilarityAnalyzer(
    method="embedding",        # Similarity method
    threshold=0.8,            # Similarity threshold
    model_name=None,          # Optional model name
    vector_store=None,        # Optional vector store
    document_processor=None,  # Optional document processor
    logging_level=logging.INFO # Logging level
)
```

## Directory Structure

```
document-similarity/
├── api/
│   ├── main.py              # FastAPI application
│   ├── routes.py            # API routes
│   └── schemas.py           # API schemas
├── models/
│   ├── document_processor.py # Document processing
│   ├── batch_analyzer.py    # Batch analysis
│   └── vector_store.py      # Vector storage
├── uploads/                 # Uploaded files
├── vector_store/           # Vector store data
├── results/                # Analysis results
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Performance Considerations

- For large datasets, consider using the batch processing endpoints
- Adjust chunk size and overlap based on document characteristics
- Choose appropriate similarity method based on use case:
  - TF-IDF: Fast but less accurate
  - Embedding: Balanced performance and accuracy
  - Sophisticated: Highest accuracy but slower

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 