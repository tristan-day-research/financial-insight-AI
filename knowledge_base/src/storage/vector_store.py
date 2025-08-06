"""
Qdrant vector store implementation
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

from knowledge_base.config.settings import get_settings

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """Simplified Qdrant vector store with client isolation."""

    def __init__(self, embeddings: Embeddings):
        self.settings = get_settings()
        self.embeddings = embeddings
        self.client = self._initialize_qdrant()
        self.collection_name = self.settings.database.qdrant_collection_name or "financial_documents"

    def _initialize_qdrant(self) -> QdrantClient:
        """Initialize Qdrant client and ensure collection exists."""
        client = QdrantClient(
            url=self.settings.database.qdrant_url,
            api_key=self.settings.database.qdrant_api_key.get_secret_value() if self.settings.database.qdrant_api_key else None,
            prefer_grpc=self.settings.database.qdrant_prefer_grpc
        )

        # Create collection if it doesn't exist
        try:
            client.get_collection(self.collection_name)
        except Exception:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Default for OpenAI embeddings
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

        return client

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Qdrant."""
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        return vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Search for similar documents."""
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        return vector_store.similarity_search(query, k=k, **kwargs)

    def delete_collection(self) -> bool:
        """Delete the entire collection (use with caution!)."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted Qdrant collection: {self.collection_name}")
        return True