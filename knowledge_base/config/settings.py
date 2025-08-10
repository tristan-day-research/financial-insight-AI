"""
Configuration settings for Financial Knowledge Base RAG System.
Centralized configuration management using Pydantic settings.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pydantic.types import SecretStr
import logging
from pathlib import Path 

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # SQLite database
    sqlite_db_path: str = Field(default="knowledge_base/data/financial_kb.db")
    
    # Qdrant Vector Database settings
    # qdrant_url: Optional[str] = Field(default=None, description="Qdrant server URL")
    # qdrant_api_key: Optional[SecretStr] = Field(default=None, description="Qdrant API key")
    qdrant_local_storage_path: str = Field(default="knowledge_base/data/qdrant_storage", description="location where Qdrant DB vectors and metadata will be store")
    qdrant_collection_name: Optional[str] = Field(default=None, description="Qdrant collection name")
    qdrant_prefer_grpc: bool = Field(default=True, description="Use gRPC for Qdrant connections")
    
    @validator("sqlite_db_path", "qdrant_local_storage_path", pre=True)
    def resolve_paths(cls, v):
        """Resolve relative paths to absolute paths."""
        if v and not Path(v).is_absolute():
            return str(PROJECT_ROOT / v)
        return v

    # @validator("vector_db_type")
    # def validate_vector_db_type(cls, v):
    #     """Validate vector database type."""
    #     valid_types = ["faiss", "chromadb", "pinecone", "qdrant"]
    #     if v not in valid_types:
    #         raise ValueError(f"Invalid vector_db_type: {v}. Must be one of {valid_types}")
    #     return v


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # OpenAI settings
    openai_api_key: Optional[SecretStr] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    fallback_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Fallback embedding model")
    
    # SEC API settings
    sec_user_agent: str = Field(default="Financial Knowledge Base RAG System", description="User agent for SEC API")
    sec_email: str = Field(default="knowledge_base@company.com", description="Email for SEC API")
    
    # Rate limiting
    max_requests_per_minute: int = Field(default=60, description="Maximum API requests per minute")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")


class DataSettings(BaseSettings):
    """Data storage and processing settings."""
    
    # Data directories
    raw_data_path: str = Field(default="knowledge_base/data/raw")
    processed_data_path: str = Field(default="knowledge_base/data/processed")
    processed_text_chunk_path: str = Field(
        default="knowledge_base/data/processed/vector_chunks",  # Direct string path
        description="Stores the text chunks external to the vector db"
    )
    output_path: str = Field(default="knowledge_base/data/outputs")

    @validator('processed_text_chunk_path', pre=True, always=True)
    def set_processed_text_chunk_path(cls, v, values):
        if v is None:
            return str(Path(values['processed_data_path']) / "vector_chunks")
        return v
    
    # Document processing for vector db chunking
    raw_docs_base_path: str = Field(default="knowledge_base/data/raw", description="Base path for raw documents")
    chunk_tokens: int = Field(default=1500, description="Target chunk size in tokens (approximated as chars for now)")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    batch_size_embed: int = Field(default=64, description="Batch size for API-based embeddings")
    api_concurrency: int = Field(default=4, description="Max concurrent API requests for embeddings")
    local_concurrency: int = Field(default=6, description="Max parallel threads for CPU-bound processing")
    batch_size_local: int = Field(default=8, description="Batch size for local document processing")
    chunk_output_path: str = Field(default="knowledge_base/data/processed/chunks.json", description="Output path for processed chunks")
    
    # File processing
    supported_formats: List[str] = Field(
        default=[".txt", ".pdf", ".docx", ".html", ".xml"],
        description="Supported file formats for processing"
    )
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    
    # Financial keywords for content analysis
    financial_keywords: List[str] = Field(
        default=[
            # Financial Statements
            "revenue", "net income", "profit", "earnings", "ebitda",
            "assets", "liabilities", "equity", "cash flow", "balance sheet",
            "income statement", "statement of operations",
            
            # Financial Metrics
            "margin", "growth", "ratio", "eps", "dividend", "yield",
            "market share", "roi", "return on investment", "roa", "roe",
            
            # Business Terms
            "sales", "costs", "expenses", "debt", "investment",
            "capital expenditure", "capex", "operating income",
            "gross profit", "net sales", "accounts receivable",
            
            # Market & Industry
            "market conditions", "competition", "industry trends",
            "regulatory", "compliance", "market share", "segment",
            
            # Risk & Performance
            "risk factors", "uncertainties", "volatility",
            "performance", "outlook", "forecast", "guidance",
            
            # Operations
            "operations", "strategy", "business model",
            "supply chain", "inventory", "research and development",
            
            # Corporate Actions
            "acquisition", "merger", "restructuring",
            "dividend", "stock repurchase", "buyback"
        ],
        description="Keywords used for financial content analysis and classification"
    )
    
    @validator("raw_data_path", "processed_data_path", "output_path", "raw_docs_base_path", pre=True)
    def resolve_data_paths(cls, v):
        """Resolve relative paths to absolute paths."""
        if v and not Path(v).is_absolute():
            # resolved = str(PROJECT_ROOT / v)
            # print(f"Resolved {v} to {resolved}")
            return str(PROJECT_ROOT / v)
        return v


class RAGSettings(BaseSettings):
    """RAG system configuration settings."""
    
    # Retrieval settings
    top_k: int = Field(default=5, description="Number of top documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    max_context_length: int = Field(default=4000, description="Maximum context length for LLM")
    
    # Prompt templates
    query_prompt_template: str = Field(
        default="""You are a financial analyst assistant. Answer the following question based on the provided context:

Context: {context}

Question: {question}

Answer:""",
        description="Template for query prompts"
    )
    
    # Cross-client settings
    enable_cross_client_search: bool = Field(default=True, description="Enable searching across multiple clients")
    cross_client_similarity_threshold: float = Field(default=0.8, description="Similarity threshold for cross-client search")


class ReportSettings(BaseSettings):
    """Report generation settings."""
    
    # Output settings
    default_output_format: str = Field(default="markdown", description="Default report output format")
    output_directory: str = Field(default="knowledge_base/data/outputs/reports")
    
    # Report types
    available_report_types: List[str] = Field(
        default=["comprehensive", "executive", "comparative", "trend"],
        description="Available report types"
    )
    
    # Report content settings
    max_sections_per_report: int = Field(default=10, description="Maximum sections per report")
    include_charts: bool = Field(default=True, description="Include charts in reports")
    include_sources: bool = Field(default=True, description="Include source citations in reports")
    
    @validator("output_directory", pre=True)
    def resolve_output_directory(cls, v):
        """Resolve relative paths to absolute paths."""
        if v and not Path(v).is_absolute():
            return str(PROJECT_ROOT / v)
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    log_file: Optional[str] = Field(default="knowledge_base/logs/app.log", description="Log file path")
    enable_console_logging: bool = Field(default=True, description="Enable console logging")
    
    @validator("log_file", pre=True)
    def resolve_log_file(cls, v):
        """Resolve relative paths to absolute paths."""
        if v and not Path(v).is_absolute():
            return str(PROJECT_ROOT / v)
        return v


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    data: DataSettings = DataSettings()
    rag: RAGSettings = RAGSettings()
    report: ReportSettings = ReportSettings()
    logging: LoggingSettings = LoggingSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance, creating it if necessary."""

    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def validate_api_keys() -> dict:
    """Validate that required API keys are available."""
    settings = get_settings()
    
    validation_results = {
        "openai_available": False,
        "pinecone_available": False,
        "sec_available": True  # SEC API doesn't require authentication
    }
    
    # Check OpenAI API key
    if settings.api.openai_api_key:
        validation_results["openai_available"] = True
    
    # Check Pinecone API key (if using Pinecone)
    if settings.database.vector_db_type == "pinecone":
        if settings.database.pinecone_api_key:
            validation_results["pinecone_available"] = True
        else:
            logging.warning("Pinecone API key not found. Please set PINECONE_API_KEY environment variable.")
    
    return validation_results


def setup_logging():
    """Setup logging configuration."""
    settings = get_settings()
    
    # Create logs directory if it doesn't exist
    if settings.logging.log_file:
        log_dir = Path(settings.logging.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.logging.log_level.upper()),
        format=settings.logging.log_format,
        handlers=[
            logging.StreamHandler() if settings.logging.enable_console_logging else None,
            logging.FileHandler(settings.logging.log_file) if settings.logging.log_file else None
        ]
    )


# Initialize logging when module is imported
setup_logging()