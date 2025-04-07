# Document Similarity Analysis

A powerful document similarity analysis tool that uses FAISS vector store and LangChain for efficient document processing and similarity search.

## Features

- **Multiple Document Types**: Support for PDF, DOCX, and TXT files
- **Advanced Similarity Methods**:
  - TF-IDF based similarity
  - Embedding-based similarity using sentence transformers
  - Sophisticated similarity with custom models
- **Efficient Processing**:
  - Document chunking with configurable size and overlap
  - Parallel processing with thread pool
  - LRU caching for improved performance
- **Vector Storage**:
  - FAISS-based vector store for efficient similarity search
  - Persistent storage with automatic saving/loading
  - Metadata management for documents
- **Robust Error Handling**:
  - Comprehensive error logging
  - Graceful fallbacks for failures
  - Input validation and type safety
- **Flexible API**:
  - RESTful endpoints for document processing
  - Background processing for large batches
  - Progress tracking and status updates

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
python main.py
```

The server will start at `http://localhost:8000`.

### API Endpoints

1. **Compare Two Documents**
```python
POST /api/compare
{
    "text1": "First document text",
    "text2": "Second document text",
    "method": "embedding",
    "threshold": 0.8
}
```

2. **Upload and Compare Files**
```python
POST /api/upload-compare
{
    "file1": <file>,
    "file2": <file>,
    "method": "embedding",
    "threshold": 0.8
}
```

3. **Process Batch of Files**
```python
POST /api/upload-batch
{
    "files": [<file1>, <file2>, ...],
    "method": "embedding",
    "threshold": 0.8,
    "model_name": "sentence-transformers/all-MiniLM-L6-v2"
}
```

4. **Compare Selected Files**
```python
POST /api/compare-selected
{
    "file_paths": ["path/to/file1", "path/to/file2"],
    "method": "embedding",
    "threshold": 0.8
}
```

5. **Get Batch Status**
```python
GET /api/batch/{batch_id}/status
```

### Python Client

```python
from batch_client import BatchDocumentClient

# Initialize client
client = BatchDocumentClient(base_url="http://localhost:8000")

# Compare selected files
response = client.compare_selected_files(
    file_paths=["path/to/file1", "path/to/file2"],
    method="embedding",
    threshold=0.8
)

# Process all files in a directory
response = client.compare_all_files(
    directory="path/to/directory",
    method="embedding",
    threshold=0.8
)

# Check batch status
status = client.get_batch_status(response.batch_id)

# Wait for completion
results = client.wait_for_completion(response.batch_id)
```

### Advanced Configuration

The application can be configured through various parameters:

1. **Document Processor**:
```python
processor = DocumentProcessor(
    upload_dir="uploads",
    supported_extensions=[".pdf", ".docx", ".txt"],
    max_workers=4,
    cache_size=1000
)
```

2. **Vector Store**:
```python
vector_store = DocumentVectorStore(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=1000,
    chunk_overlap=200,
    store_path="vector_store"
)
```

3. **Batch Analyzer**:
```python
analyzer = BatchSimilarityAnalyzer(
    method="embedding",
    threshold=0.8,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_workers=4,
    cache_size=1000
)
```

## Directory Structure

```
document-similarity/
├── api/
│   ├── routes.py
│   └── schemas.py
├── models/
│   ├── batch_analyzer.py
│   ├── document_processor.py
│   └── vector_store.py
├── batch_client.py
├── main.py
├── requirements.txt
└── README.md
```

## Performance Considerations

- Use appropriate chunk sizes based on document types
- Adjust thread pool size based on available resources
- Configure cache size based on memory constraints
- Choose similarity method based on accuracy vs. speed requirements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 