from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import asyncio

from .document_processor import DocumentProcessor
from .vector_store import DocumentVectorStore


class BatchSimilarityAnalyzer:
    """Analyzes similarity between multiple documents in batch."""
    
    def __init__(
        self,
        method: str = "embedding",
        threshold: float = 0.8,
        model_name: Optional[str] = None,
        vector_store: Optional[DocumentVectorStore] = None,
        document_processor: Optional[DocumentProcessor] = None,
        logging_level: int = logging.INFO
    ):
        """
        Initialize the batch analyzer.
        
        Args:
            method: Similarity method ('tfidf', 'embedding', or 'sophisticated')
            threshold: Similarity threshold for considering documents as duplicates
            model_name: Name of the sentence transformer model to use
            vector_store: Optional custom vector store
            document_processor: Optional custom document processor
            logging_level: Logging level
        """
        # Configure logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        
        self.method = method.lower()
        self.threshold = threshold
        self.model_name = model_name
        
        # Initialize document processor and vector store
        self.vector_store = vector_store or DocumentVectorStore(
            embedding_model=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.document_processor = document_processor or DocumentProcessor(
            vector_store=self.vector_store
        )
        
        # Initialize TF-IDF vectorizer if needed
        if self.method == "tfidf":
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english'
            )
        
        # Initialize models
        if self.method == "embedding":
            self.embedding_model = SentenceTransformer(
                model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
    
    async def _compute_tfidf_similarity(self, texts: List[str]) -> np.ndarray:
        """Compute similarity matrix using TF-IDF."""
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
        except Exception as e:
            self.logger.error(f"Error computing TF-IDF similarity: {e}")
            return np.zeros((len(texts), len(texts)))
    
    async def _compute_embedding_similarity(self, texts: List[str]) -> np.ndarray:
        """Compute similarity using embedding model."""
        try:
            embeddings = self.embedding_model.encode(texts)
            return cosine_similarity(embeddings)
        except Exception as e:
            self.logger.error(f"Error computing embedding similarity: {e}")
            return np.zeros((len(texts), len(texts)))
    
    async def _compute_sophisticated_similarity(
        self,
        query: str,
        doc_ids: List[str],
        **kwargs
    ) -> List[Dict]:
        """Compute similarity using sophisticated embedding model."""
        try:
            # Use a more advanced model for sophisticated comparison
            if self.model_name:
                self.vector_store.embeddings = SentenceTransformer(
                    model_name=self.model_name
                )
            
            return self.document_processor.find_similar_documents(
                query=query,
                threshold=self.threshold,
                doc_ids=doc_ids,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error computing sophisticated similarity: {e}")
            return []
    
    async def find_duplicates(
        self,
        documents: Dict[str, str],
        similarity_callback: Optional[Callable] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Find duplicate or highly similar documents.
        
        Args:
            documents: Dictionary of document IDs to file paths
            similarity_callback: Optional callback function for custom similarity processing
            **kwargs: Additional arguments for similarity search
            
        Returns:
            List of dictionaries containing duplicate pairs and their similarity scores
        """
        try:
            duplicates = []
            doc_ids = list(documents.keys())
            
            if self.method == "tfidf":
                # Get all document texts
                texts = []
                for doc_id in doc_ids:
                    chunks = self.document_processor.get_document_chunks(doc_id)
                    if chunks:
                        texts.append(" ".join(chunk["content"] for chunk in chunks))
                    else:
                        texts.append("")
                
                # Compute similarity matrix
                similarity_matrix = await self._compute_tfidf_similarity(texts)
                
                # Find duplicates
                for i in range(len(doc_ids)):
                    for j in range(i + 1, len(doc_ids)):
                        if similarity_matrix[i, j] >= self.threshold:
                            duplicates.append({
                                'file1': doc_ids[i],
                                'file2': doc_ids[j],
                                'similarity': float(similarity_matrix[i, j]),
                                'metadata': {
                                    'file1_metadata': self.document_processor.vector_store.get_document_metadata(doc_ids[i]),
                                    'file2_metadata': self.document_processor.vector_store.get_document_metadata(doc_ids[j])
                                }
                            })
            
            else:
                # Compare each document with others using embedding methods
                for doc_id1 in doc_ids:
                    # Get document chunks
                    chunks1 = self.document_processor.get_document_chunks(doc_id1)
                    if not chunks1:
                        continue
                    
                    # Use first chunk as query
                    query = chunks1[0]["content"]
                    
                    # Find similar documents based on method
                    if self.method == "embedding":
                        similar_docs = await self._compute_embedding_similarity([query])
                    else:  # sophisticated
                        similar_docs = await self._compute_sophisticated_similarity(
                            query=query,
                            doc_ids=doc_ids,
                            **kwargs
                        )
                    
                    # Process results with callback if provided
                    if similarity_callback:
                        similar_docs = similarity_callback(similar_docs, doc_id1)
                    
                    # Add to duplicates
                    for doc in similar_docs:
                        if doc["doc_id"] != doc_id1:  # Avoid self-matches
                            duplicates.append({
                                'file1': doc_id1,
                                'file2': doc["doc_id"],
                                'similarity': doc["score"],
                                'metadata': {
                                    'file1_metadata': self.document_processor.vector_store.get_document_metadata(doc_id1),
                                    'file2_metadata': self.document_processor.vector_store.get_document_metadata(doc["doc_id"])
                                }
                            })
            
            return duplicates
        except Exception as e:
            self.logger.error(f"Error finding duplicates: {e}")
            return []
    
    async def analyze_directory(
        self,
        directory: Union[str, Path],
        output_file: Optional[str] = None,
        file_filter: Optional[Callable] = None,
        custom_metadata: Optional[Dict] = None,
        force_update: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Analyze a directory of documents.
        
        Args:
            directory: Path to directory containing documents
            output_file: Optional path to save results
            file_filter: Optional function to filter files
            custom_metadata: Optional metadata to add to documents
            force_update: Whether to force reprocessing of documents
            **kwargs: Additional arguments for document processing
            
        Returns:
            DataFrame containing duplicate pairs and their similarity scores
        """
        try:
            directory = Path(directory)
            self.logger.info(f"Analyzing documents in {directory} using {self.method} method...")
            
            # Process documents
            documents = self.document_processor.process_directory(
                directory,
                file_filter=file_filter,
                custom_metadata=custom_metadata,
                force_update=force_update,
                **kwargs
            )
            
            # Find duplicates
            duplicates = await self.find_duplicates(documents, **kwargs)
            
            # Convert to DataFrame
            df = pd.DataFrame(duplicates)
            
            if output_file:
                df.to_csv(output_file, index=False)
                self.logger.info(f"Results saved to {output_file}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error analyzing directory {directory}: {e}")
            return pd.DataFrame()
    
    async def analyze_selected_files(
        self,
        file_paths: List[Union[str, Path]],
        output_file: Optional[str] = None,
        custom_metadata: Optional[Dict] = None,
        force_update: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Analyze selected files.
        
        Args:
            file_paths: List of file paths to analyze
            output_file: Optional path to save results
            custom_metadata: Optional metadata to add to documents
            force_update: Whether to force reprocessing of documents
            **kwargs: Additional arguments for document processing
            
        Returns:
            DataFrame containing duplicate pairs and their similarity scores
        """
        try:
            self.logger.info(f"Analyzing {len(file_paths)} files using {self.method} method...")
            
            # Process documents
            documents = self.document_processor.process_selected_files(
                [Path(p) for p in file_paths],
                custom_metadata=custom_metadata,
                force_update=force_update,
                **kwargs
            )
            
            # Find duplicates
            duplicates = await self.find_duplicates(documents, **kwargs)
            
            # Convert to DataFrame
            df = pd.DataFrame(duplicates)
            
            if output_file:
                df.to_csv(output_file, index=False)
                self.logger.info(f"Results saved to {output_file}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error analyzing selected files: {e}")
            return pd.DataFrame()
    
    async def get_similarity_summary(
        self,
        df: pd.DataFrame,
        additional_metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate a summary of the similarity analysis.
        
        Args:
            df: DataFrame containing similarity results
            additional_metrics: Optional list of additional metrics to compute
            
        Returns:
            Dictionary containing summary statistics
        """
        try:
            if df.empty:
                return {
                    'total_documents': 0,
                    'duplicate_pairs': 0,
                    'max_similarity': 0.0,
                    'avg_similarity': 0.0
                }
            
            summary = {
                'total_documents': len(df),
                'duplicate_pairs': len(df),
                'max_similarity': float(df['similarity'].max()),
                'avg_similarity': float(df['similarity'].mean()),
                'min_similarity': float(df['similarity'].min()),
                'std_similarity': float(df['similarity'].std())
            }
            
            # Add custom metrics if provided
            if additional_metrics:
                for metric in additional_metrics:
                    if metric in df.columns:
                        summary[metric] = float(df[metric].mean())
            
            return summary
        except Exception as e:
            self.logger.error(f"Error generating similarity summary: {e}")
            return {}
    
    async def update_document_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """Update metadata for a specific document."""
        try:
            return self.document_processor.update_document_metadata(doc_id, metadata)
        except Exception as e:
            self.logger.error(f"Error updating metadata for {doc_id}: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store."""
        try:
            return self.document_processor.delete_document(doc_id)
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    async def get_stats(self) -> Dict:
        """Get statistics about the analyzer."""
        try:
            return self.document_processor.get_stats()
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}
    
    async def cleanup(self, max_age: Optional[int] = None):
        """Clean up old documents."""
        try:
            self.document_processor.cleanup(max_age)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") 