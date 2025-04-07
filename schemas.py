from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class SimilarityMethod(str, Enum):
    """Enum for similarity methods."""
    TFIDF = "tfidf"  # Basic TF-IDF based similarity
    EMBEDDING = "embedding"  # Sentence transformer embeddings
    SOPHISTICATED = "sophisticated"  # More sophisticated embedding model


class SimilarityRequest(BaseModel):
    """Request model for comparing two text documents."""
    text1: str
    text2: str
    method: SimilarityMethod = SimilarityMethod.TFIDF
    threshold: float = Field(0.8, ge=0.0, le=1.0)


class SimilarityResponse(BaseModel):
    """Response model for document similarity comparison."""
    similarity: float = Field(..., ge=0.0, le=1.0)


class BatchUploadResponse(BaseModel):
    """Response model for batch upload."""
    batch_id: str
    status: str


class BatchStatus(BaseModel):
    """Model for batch processing status."""
    status: str
    progress: Optional[int] = None
    error: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


class BatchResults(BaseModel):
    """Model for batch processing results."""
    file1: str
    file2: str
    similarity: float = Field(..., ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str 