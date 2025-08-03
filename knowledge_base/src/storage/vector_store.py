"""
Vector database interface for storing and retrieving document embeddings.
Supports FAISS, ChromaDB, and Pinecone with client-specific filtering.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
import numpy as np
import json
import pickle
from datetime import datetime
import hashlib

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config.settings import get_settings, validate_api_keys

# Vector store imports
import faiss
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# LangChain imports
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores with client isolation."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document], client_id: str) -> List[str]:
        """Add documents for a specific client."""
        pass
    
    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        client_id: Optional[str] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """Search with optional client filtering."""
        pass
    
    @abstractmethod
    def delete_client_data(self, client_id: str) -> bool:
        """Delete all data for a specific client."""
        pass


class FinancialVectorStore:
    """
    Main vector store class with client isolation and federation capabilities.
    Supports both FAISS and ChromaDB backends.
    """
    
    def __init__(self, store_type: str = "faiss"):
        self.settings = get_settings()
        self.store_type = store_type.lower()
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Client metadata tracking
        self.client_metadata = {}
        self._load_client_metadata()
    
    def _initialize_embeddings(self):
        """Initialize embedding model with fallback support."""
        api_validation = validate_api_keys()
        
        if api_validation["openai_available"]:
            try:
                embeddings = OpenAIEmbeddings(
                    model=self.settings.api.embedding_model,
                    openai_api_key=self.settings.api.openai_api_key
                )
                logger.info(f"Using OpenAI embeddings: {self.settings.api.embedding_model}")
                return embeddings
            except Exception as e:
                logger.warning(f"OpenAI embeddings failed: {e}. Falling back to local model.")
        
        # Fallback to local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.api.fallback_embedding_model
        )
        logger.info(f"Using local embeddings: {self.settings.api.fallback_embedding_model}")
        return embeddings
    
    def _initialize_vector_store(self):
        """Initialize the appropriate vector store backend."""
        if self.store_type == "faiss":
            return self._initialize_faiss()
        elif self.store_type == "chromadb" and CHROMADB_AVAILABLE:
            return self._initialize_chromadb()
        else:
            logger.warning(f"Store type {self.store_type} not available. Falling back to FAISS.")
            return self._initialize_faiss()
    
    def _initialize_faiss(self):
        """Initialize FAISS vector store."""
        index_path = Path(self.settings.database.faiss_index_path)
        
        if index_path.exists() and (index_path / "index.faiss").exists():
            try:
                # Load existing FAISS index
                vector_store = FAISS.load_local(
                    str(index_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing FAISS index from {index_path}")
                return vector_store
            except Exception as e:
                logger.warning(f"Failed to load existing FAISS index: {e}")
        
        # Create new FAISS index with a dummy document
        dummy_doc = Document(page_content="initialization", metadata={"type": "init"})
        vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
        
        # Save the index
        index_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_path))
        
        logger.info(f"Created new FAISS index at {index_path}")
        return vector_store
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB vector store."""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available")
        
        persist_directory = self.settings.database.chromadb_path
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        collection_name = "financial_documents"
        try:
            collection = client.get_collection(collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except:
            collection = client.create_collection(collection_name)
            logger.info(f"Created new ChromaDB collection: {collection_name}")
        
        # Create LangChain ChromaDB wrapper
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        return vector_store
    
    def add_documents(
        self, 
        documents: List[Document], 
        client_id: str,
        batch_size: int = 50
    ) -> List[str]:
        """
        Add documents to the vector store with client metadata.
        
        Args:
            documents: List of LangChain documents
            client_id: Client identifier for isolation
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Add client_id to all document metadata
        for doc in documents:
            doc.metadata["client_id"] = client_id
            doc.metadata["indexed_date"] = np.datetime64('now').isoformat()
        
        document_ids = []
        
        # Process in batches to handle large document sets
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                if self.store_type == "faiss":
                    # FAISS doesn't return IDs directly, so we'll generate them
                    batch_ids = [f"{client_id}_{i + j}" for j in range(len(batch))]
                    
                    # Add IDs to metadata
                    for doc, doc_id in zip(batch, batch_ids):
                        doc.metadata["document_id"] = doc_id
                    
                    # Add to FAISS store
                    if hasattr(self.vector_store, 'docstore') and len(self.vector_store.docstore._dict) > 1:
                        # Existing store with documents
                        self.vector_store.add_documents(batch)
                    else:
                        # First real documents (remove dummy if exists)
                        new_store = FAISS.from_documents(batch, self.embeddings)
                        self.vector_store = new_store
                    
                    document_ids.extend(batch_ids)
                    
                else:  # ChromaDB
                    batch_ids = self.vector_store.add_documents(batch)
                    document_ids.extend(batch_ids)
                
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents for client {client_id}")
                
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1} for client {client_id}: {str(e)}")
                continue
        
        # Update client metadata
        self._update_client_metadata(client_id, len(documents))
        
        # Save the updated index
        self._save_vector_store()
        
        logger.info(f"Successfully added {len(document_ids)} documents for client {client_id}")
        return document_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        client_id: Optional[str] = None,
        enable_cross_client: bool = False,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search with optional client filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            client_id: Optional client ID for filtering
            enable_cross_client: Allow cross-client search for federation
            filter_metadata: Additional metadata filters
            
        Returns:
            List of similar documents
        """
        try:
            # Build metadata filter
            search_filter = filter_metadata.copy() if filter_metadata else {}
            
            # Apply client isolation if specified
            if client_id and not enable_cross_client:
                search_filter["client_id"] = client_id
            
            # Perform search based on store type
            if self.store_type == "faiss":
                # FAISS search with post-filtering
                # Get more results than needed for filtering
                search_k = k * 3 if search_filter else k
                
                results = self.vector_store.similarity_search(query, k=search_k)
                
                # Apply metadata filtering
                if search_filter:
                    filtered_results = []
                    for doc in results:
                        if self._matches_filter(doc.metadata, search_filter):
                            filtered_results.append(doc)
                        if len(filtered_results) >= k:
                            break
                    results = filtered_results
                else:
                    results = results[:k]
                    
            else:  # ChromaDB
                # ChromaDB supports native metadata filtering
                where = self._build_chromadb_filter(search_filter) if search_filter else None
                results = self.vector_store.similarity_search(
                    query, 
                    k=k, 
                    where=where
                )
            
            # Log search details
            search_type = "client-specific" if client_id and not enable_cross_client else "federated"
            logger.info(f"Performed {search_type} search: {len(results)} results for query '{query[:50]}...'")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if document metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def _build_chromadb_filter(self, filter_dict: Dict) -> Dict:
        """Build ChromaDB where clause from filter dictionary."""
        where = {}
        for key, value in filter_dict.items():
            if isinstance(value, list):
                where[key] = {"$in": value}
            else:
                where[key] = value
        return where
    
    def get_client_statistics(self, client_id: str) -> Dict:
        """Get statistics for a specific client."""
        stats = {
            "client_id": client_id,
            "document_count": 0,
            "last_updated": None,
            "index_size": 0
        }
        
        if client_id in self.client_metadata:
            stats.update(self.client_metadata[client_id])
        
        return stats
    
    def list_clients(self) -> List[str]:
        """Get list of all clients in the vector store."""
        return list(self.client_metadata.keys())
    
    def delete_client_data(self, client_id: str) -> bool:
        """
        Delete all data for a specific client.
        Note: This is a complex operation and may require rebuilding the index.
        """
        try:
            # For FAISS, we'd need to rebuild the entire index without client data
            # For ChromaDB, we can delete by metadata filter
            
            if self.store_type == "chromadb":
                # Delete documents with matching client_id
                collection = self.vector_store._collection
                collection.delete(where={"client_id": client_id})
                logger.info(f"Deleted all data for client {client_id} from ChromaDB")
            else:
                # For FAISS, this would require a more complex rebuild
                logger.warning(f"Client data deletion for FAISS requires manual rebuild")
                return False
            
            # Remove from client metadata
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]
                self._save_client_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting client data for {client_id}: {str(e)}")
            return False
    
    def _update_client_metadata(self, client_id: str, doc_count: int):
        """Update metadata for a client."""
        if client_id not in self.client_metadata:
            self.client_metadata[client_id] = {
                "document_count": 0,
                "first_indexed": np.datetime64('now').isoformat(),
                "last_updated": None
            }
        
        self.client_metadata[client_id]["document_count"] += doc_count
        self.client_metadata[client_id]["last_updated"] = np.datetime64('now').isoformat()
        
        self._save_client_metadata()
    
    def _load_client_metadata(self):
        """Load client metadata from disk."""
        metadata_file = Path(self.settings.database.faiss_index_path) / "client_metadata.pkl"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    self.client_metadata = pickle.load(f)
                logger.info(f"Loaded metadata for {len(self.client_metadata)} clients")
            except Exception as e:
                logger.warning(f"Failed to load client metadata: {e}")
                self.client_metadata = {}
        else:
            self.client_metadata = {}
    
    def _save_client_metadata(self):
        """Save client metadata to disk."""
        metadata_file = Path(self.settings.database.faiss_index_path) / "client_metadata.pkl"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.client_metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save client metadata: {e}")
    
    def _save_vector_store(self):
        """Save vector store to disk."""
        if self.store_type == "faiss":
            try:
                index_path = Path(self.settings.database.faiss_index_path)
                self.vector_store.save_local(str(index_path))
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")
        # ChromaDB auto-persists


def main():
    """Example usage of vector store."""
    # Initialize vector store
    vector_store = FinancialVectorStore(store_type="faiss")
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="Apple Inc. reported revenue of $394.3 billion for fiscal year 2023.",
            metadata={"ticker": "AAPL", "filing_type": "10-K", "section": "financial_data"}
        ),
        Document(
            page_content="Microsoft Corporation's net income increased by 20% year over year.",
            metadata={"ticker": "MSFT", "filing_type": "10-Q", "section": "financial_data"}
        )
    ]
    
    # Add documents for different clients
    vector_store.add_documents(sample_docs[:1], client_id="AAPL")
    vector_store.add_documents(sample_docs[1:], client_id="MSFT")
    
    # Perform searches
    print("\n--- Client-specific search (AAPL only) ---")
    results = vector_store.similarity_search("revenue", k=2, client_id="AAPL")
    for doc in results:
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Client: {doc.metadata.get('client_id')}")
    
    print("\n--- Federated search (all clients) ---")
    results = vector_store.similarity_search("revenue", k=2, enable_cross_client=True)
    for doc in results:
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Client: {doc.metadata.get('client_id')}")
    
    # Show statistics
    print(f"\nClients in store: {vector_store.list_clients()}")
    for client in vector_store.list_clients():
        stats = vector_store.get_client_statistics(client)
        print(f"{client}: {stats['document_count']} documents")


if __name__ == "__main__":
    main()