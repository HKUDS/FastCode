"""
Code Embedder - Generate embeddings for code snippets
"""

import logging
import platform
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class CodeEmbedder:
    """Generate embeddings for code using sentence transformers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_config = config.get("embedding", {})
        self.logger = logging.getLogger(__name__)
        
        self.model_name = self.embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = self.embedding_config.get("device", "cpu")
        self.batch_size = self.embedding_config.get("batch_size", 32)
        self.max_seq_length = self.embedding_config.get("max_seq_length", 512)
        self.normalize = self.embedding_config.get("normalize_embeddings", True)
        
        # Check for CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.logger.info(f"Loading embedding model: {self.model_name}")
        self.model = self._load_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model"""
        model = SentenceTransformer(self.model_name, device=self.device)
        model.max_seq_length = self.max_seq_length
        return model
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts
        
        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
        
        encode_kwargs = {
            'batch_size': self.batch_size,
            'show_progress_bar': len(texts) > 100,
            'normalize_embeddings': self.normalize,
            'convert_to_numpy': True,
            'device': self.device,
            'convert_to_tensor': False,
        }
        
        if platform.system() == 'Darwin':
            encode_kwargs['pool'] = None
        
        embeddings = self.model.encode(texts, **encode_kwargs)
        
        return embeddings
    
    def embed_code_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for code elements (functions, classes, etc.)
        
        Args:
            elements: List of code element dictionaries
        
        Returns:
            List of elements with embeddings added
        """
        if not elements:
            return []
        
        # Prepare texts for embedding
        texts = [self._prepare_code_text(elem) for elem in elements]
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(texts)} code elements")
        embeddings = self.embed_batch(texts)
        self.logger.info(f"✓ Successfully generated embeddings for {len(embeddings)} code elements")
        
        # Add embeddings to elements
        for elem, embedding in zip(elements, embeddings):
            elem["embedding"] = embedding
            elem["embedding_text"] = texts[elements.index(elem)]
        
        return elements
    
    def _prepare_code_text(self, element: Dict[str, Any]) -> str:
        """
        Prepare code element for embedding
        
        Combines various parts of the code element into a single text
        suitable for embedding
        """
        parts = []
        
        # Add type
        if "type" in element:
            parts.append(f"Type: {element['type']}")
        
        # Add name
        if "name" in element:
            parts.append(f"Name: {element['name']}")
        
        # Add signature (for functions)
        if "signature" in element:
            parts.append(f"Signature: {element['signature']}")
        
        # Add docstring/description
        if "docstring" in element and element["docstring"]:
            parts.append(f"Documentation: {element['docstring']}")
        
        # Add summary
        if "summary" in element and element["summary"]:
            parts.append(element["summary"])
        
        # Add code snippet (truncated)
        if "code" in element:
            code = element["code"]
            if len(code) > 10000:  # Truncate long code
                code = code[:10000] + "..."
            parts.append(f"Code:\n{code}")
        
        return "\n".join(parts)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Similarity score (0-1)
        """
        if self.normalize:
            # Already normalized, just dot product
            return float(np.dot(embedding1, embedding2))
        else:
            # Compute cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                            embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarities between query and multiple embeddings
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embedding vectors
        
        Returns:
            Array of similarity scores
        """
        if self.normalize:
            # Simple dot product for normalized embeddings
            similarities = np.dot(embeddings, query_embedding)
        else:
            # Compute cosine similarities
            norms = np.linalg.norm(embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return np.zeros(len(embeddings))
            similarities = np.dot(embeddings, query_embedding) / (norms * query_norm)
        
        return similarities

