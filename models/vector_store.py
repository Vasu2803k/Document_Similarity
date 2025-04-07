from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import pickle
import os
import logging
import hashlib
import time
from sentence_transformers import SentenceTransformer
import json
import aiofiles
import asyncio


class DocumentVectorStore:
    """Manages document storage and similarity search using FAISS."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        store_path: str = "vector_store",
        dimension: int = 384,
        similarity_metric: str = "cosine",
        max_documents: int = 1000000,
        logging_level: int = logging.INFO
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: Name of the sentence transformer model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            store_path: Path to store vectors and metadata
            dimension: Dimension of the embedding vectors
            similarity_metric: Metric for similarity search
            max_documents: Maximum number of documents to store
            logging_level: Logging level
        """
        # Configure logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        
        # Initialize parameters
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store_path = Path(store_path)
        self.dimension = dimension
        self.max_documents = max_documents
        
        # Create store directory
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize embedding model
        self.embeddings = SentenceTransformer(embedding_model)
        
        # Initialize FAISS index
        if similarity_metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        # Initialize document storage
        self.documents: Dict[str, List[Dict]] = {}
        self.metadata: Dict[str, Dict] = {}
        self.chunk_to_doc: Dict[int, str] = {}
        
        # Load existing store if available
        asyncio.create_task(self.load())
    
    async def _create_documents(self, text: str, metadata: Dict) -> List[Document]:
        """Create LangChain documents from text."""
        try:
            docs = self.text_splitter.create_documents([text])
            for doc in docs:
                doc.metadata.update(metadata)
            return docs
        except Exception as e:
            self.logger.error(f"Error creating documents: {e}")
            return []
    
    def _get_document_hash(self, text: str) -> str:
        """Generate a hash for a document."""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None,
        custom_chunks: Optional[List[str]] = None,
        force_update: bool = False
    ) -> bool:
        """
        Add a document to the vector store.
        
        Args:
            doc_id: Unique identifier for the document
            text: Document text
            metadata: Optional document metadata
            custom_chunks: Optional custom text chunks
            force_update: Whether to force update existing document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if document exists and update is not forced
            if not force_update and doc_id in self.documents:
                self.logger.info(f"Document {doc_id} already exists")
                return True
            
            # Prepare metadata
            doc_metadata = {
                "doc_id": doc_id,
                "added_at": time.time(),
                **(metadata or {})
            }
            
            # Create or use custom chunks
            if custom_chunks:
                docs = [Document(page_content=chunk, metadata=doc_metadata) 
                       for chunk in custom_chunks]
            else:
                docs = await self._create_documents(text, doc_metadata)
            
            if not docs:
                return False
            
            # Get embeddings for chunks
            chunk_texts = [doc.page_content for doc in docs]
            chunk_embeddings = self.embeddings.encode(chunk_texts)
            
            # Add to FAISS index
            self.index.add(np.array(chunk_embeddings).astype('float32'))
            
            # Store document chunks and metadata
            self.documents[doc_id] = [
                {
                    "content": doc.page_content,
                    "embedding": embedding.tolist()
                }
                for doc, embedding in zip(docs, chunk_embeddings)
            ]
            
            # Update metadata
            self.metadata[doc_id] = doc_metadata
            
            # Update chunk to document mapping
            start_idx = len(self.chunk_to_doc)
            for i in range(len(docs)):
                self.chunk_to_doc[start_idx + i] = doc_id
            
            # Save store
            await self.save()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document {doc_id}: {e}")
            return False
    
    async def similarity_search(
        self,
        query: Union[str, List[str]],
        k: int = 5,
        threshold: float = 0.8,
        doc_ids: Optional[List[str]] = None,
        filter_metadata: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Query text or list of texts
            k: Number of results to return
            threshold: Similarity threshold
            doc_ids: Optional list of document IDs to search in
            filter_metadata: Optional metadata filters
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Get query embedding
            if isinstance(query, str):
                query = [query]
            query_embedding = self.embeddings.encode(query)
            
            # Search in FAISS index
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'),
                min(k * 2, self.index.ntotal)
            )
            
            # Process results
            results = []
            seen_docs = set()
            
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx in self.chunk_to_doc:
                    doc_id = self.chunk_to_doc[idx]
                    
                    # Skip if document ID is filtered
                    if doc_ids and doc_id not in doc_ids:
                        continue
                    
                    # Skip if document is already in results
                    if doc_id in seen_docs:
                        continue
                    
                    # Check metadata filters
                    if filter_metadata:
                        doc_metadata = self.metadata.get(doc_id, {})
                        if not all(doc_metadata.get(k) == v 
                                 for k, v in filter_metadata.items()):
                            continue
                    
                    # Calculate similarity score
                    similarity = 1.0 / (1.0 + dist)
                    
                    if similarity >= threshold:
                        results.append({
                            "doc_id": doc_id,
                            "score": float(similarity),
                            "metadata": self.metadata.get(doc_id, {})
                        })
                        seen_docs.add(doc_id)
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    async def get_document_chunks(
        self,
        doc_id: str,
        chunk_indices: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Get chunks for a specific document.
        
        Args:
            doc_id: Document ID
            chunk_indices: Optional list of chunk indices
            
        Returns:
            List of document chunks
        """
        try:
            if doc_id not in self.documents:
                return []
            
            chunks = self.documents[doc_id]
            if chunk_indices is not None:
                chunks = [chunks[i] for i in chunk_indices 
                         if 0 <= i < len(chunks)]
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error getting chunks for {doc_id}: {e}")
            return []
    
    async def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a specific document."""
        return self.metadata.get(doc_id)
    
    async def update_document_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """Update metadata for a specific document."""
        try:
            if doc_id in self.metadata:
                self.metadata[doc_id].update(metadata)
                await self.save()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating metadata for {doc_id}: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store."""
        try:
            if doc_id in self.documents:
                # Remove document chunks
                del self.documents[doc_id]
                
                # Remove metadata
                del self.metadata[doc_id]
                
                # Remove from chunk mapping
                self.chunk_to_doc = {
                    idx: doc_id for idx, doc_id in self.chunk_to_doc.items()
                    if doc_id != doc_id
                }
                
                # Save changes
                await self.save()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    async def save(self):
        """Save the vector store to disk."""
        try:
            # Save FAISS index
            faiss.write_index(
                self.index,
                str(self.store_path / "index.faiss")
            )
            
            # Save documents and metadata
            store_data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "chunk_to_doc": self.chunk_to_doc
            }
            
            async with aiofiles.open(self.store_path / "store.pkl", "wb") as f:
                await f.write(pickle.dumps(store_data))
                
        except Exception as e:
            self.logger.error(f"Error saving vector store: {e}")
    
    async def load(self):
        """Load the vector store from disk."""
        try:
            index_path = self.store_path / "index.faiss"
            store_path = self.store_path / "store.pkl"
            
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            
            if store_path.exists():
                async with aiofiles.open(store_path, "rb") as f:
                    store_data = pickle.load(f)
                    self.documents = store_data.get("documents", {})
                    self.metadata = store_data.get("metadata", {})
                    self.chunk_to_doc = store_data.get("chunk_to_doc", {})
                    
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": sum(len(chunks) for chunks in self.documents.values()),
            "index_size": self.index.ntotal if self.index else 0,
            "store_size": sum(
                os.path.getsize(f) for f in self.store_path.glob("*")
                if f.is_file()
            )
        }
    
    async def cleanup(self, max_age: Optional[int] = None):
        """Clean up old documents."""
        try:
            if max_age:
                current_time = time.time()
                for doc_id, metadata in list(self.metadata.items()):
                    if current_time - metadata.get("added_at", 0) > max_age:
                        await self.delete_document(doc_id)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") 