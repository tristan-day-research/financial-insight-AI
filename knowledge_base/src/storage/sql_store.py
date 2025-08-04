"""
SQL database interface for storing metadata and document information.
Provides client management, document tracking, and analytics capabilities.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import sqlite3
import json
from datetime import datetime
import pandas as pd

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config.settings import get_settings

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import Index

logger = logging.getLogger(__name__)

Base = declarative_base()


class Client(Base):
    """Client information table."""
    __tablename__ = "clients"
    
    id = Column(String(20), primary_key=True)  # Ticker symbol or client ID
    company_name = Column(String(200))
    cik = Column(String(20))  # SEC CIK number
    industry = Column(String(100))
    sector = Column(String(100))
    market_cap = Column(Float)
    created_date = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="client")
    financial_metrics = relationship("FinancialMetric", back_populates="client")
    
    def __repr__(self):
        return f"<Client(id='{self.id}', name='{self.company_name}')>"


class Document(Base):
    """Document metadata table."""
    __tablename__ = "documents"
    
    document_pk = Column(Integer, primary_key=True, autoincrement=True)  # Internal primary key
    document_id = Column(String(100), unique=True, index=True)  # Unique document identifier (SEC accession number)
    client_id = Column(String(20), ForeignKey("clients.id"), index=True)
    
    # Document information
    filing_type = Column(String(20))  # 10-K, 10-Q, 8-K, etc.
    filing_date = Column(String(20))
    period_of_report = Column(String(20))
    fiscal_year_end = Column(String(10))
    
    # File information
    file_path = Column(Text)
    file_size = Column(Integer)
    download_date = Column(DateTime)
    processed_date = Column(DateTime)
    
    # Content analysis
    total_chunks = Column(Integer, default=0)
    financial_density = Column(Float, default=0.0)
    has_revenue_data = Column(Boolean, default=False)
    has_profit_data = Column(Boolean, default=False)
    has_balance_sheet = Column(Boolean, default=False)
    has_cash_flow = Column(Boolean, default=False)
    
    # Additional metadata
    metadata_json = Column(JSON)  # Store additional metadata as JSON
    
    # Relationships
    client = relationship("Client", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")
    
    # Indexes
    __table_args__ = (
        Index('idx_client_filing', 'client_id', 'filing_type'),
        Index('idx_filing_date', 'filing_date'),
    )
    
    def __repr__(self):
        return f"<Document(document_id='{self.document_id}', client='{self.client_id}', type='{self.filing_type}')>"


class DocumentChunk(Base):
    """Individual document chunks table."""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_pk = Column(Integer, ForeignKey("documents.document_pk"), index=True)  # Reference to document primary key
    chunk_index = Column(Integer)
    
    # Chunk content metadata
    section = Column(String(100))
    content_type = Column(String(50))  # financial_statements, risk_analysis, etc.
    chunk_size = Column(Integer)
    financial_density = Column(Float, default=0.0)
    
    # Content indicators
    contains_numbers = Column(Boolean, default=False)
    contains_financial_terms = Column(Boolean, default=False)
    key_topics = Column(JSON)  # Store as JSON array
    
    # Vector store reference
    vector_id = Column(String(100))  # Reference to vector store document ID
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_chunk', 'document_pk', 'chunk_index'),
        Index('idx_section', 'section'),
        Index('idx_content_type', 'content_type'),
    )


class FinancialMetric(Base):
    """Extracted financial metrics table."""
    __tablename__ = "financial_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(20), ForeignKey("clients.id"), index=True)
    document_pk = Column(Integer, ForeignKey("documents.document_pk"))  # Reference to document primary key
    
    # Metric information
    metric_name = Column(String(100), index=True)  # revenue, net_income, etc.
    metric_value = Column(Float)
    metric_unit = Column(String(20))  # USD, millions, etc.
    
    # Time period
    period_type = Column(String(20))  # annual, quarterly
    period_end_date = Column(String(20))
    fiscal_year = Column(Integer, index=True)
    fiscal_quarter = Column(Integer)
    
    # Source information
    source_section = Column(String(100))
    extraction_confidence = Column(Float, default=0.0)
    extraction_method = Column(String(50))  # manual, regex, llm, etc.
    
    # Additional context
    context_text = Column(Text)  # Surrounding text for context
    metadata_json = Column(JSON)
    
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    client = relationship("Client", back_populates="financial_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_client_metric', 'client_id', 'metric_name'),
        Index('idx_fiscal_period', 'fiscal_year', 'fiscal_quarter'),
        Index('idx_period_end', 'period_end_date'),
    )


class FinancialSQLStore:
    """SQL database manager for financial data."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._create_tables()
    
    def _create_engine(self):
        """Create SQLAlchemy engine."""
        db_path = Path(self.settings.database.sqlite_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        database_url = f"sqlite:///{db_path}"
        engine = create_engine(database_url, echo=False)
        
        logger.info(f"Created SQL engine: {database_url}")
        return engine
    
    def _create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Created SQL tables")
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def add_client(self, client_data: Dict) -> bool:
        """Add or update client information."""
        session = self.get_session()
        try:
            # Check if client exists
            existing_client = session.query(Client).filter(Client.id == client_data["id"]).first()
            
            if existing_client:
                # Update existing client
                for key, value in client_data.items():
                    if hasattr(existing_client, key):
                        setattr(existing_client, key, value)
                existing_client.last_updated = datetime.utcnow()
            else:
                # Create new client
                client = Client(**client_data)
                session.add(client)
            
            session.commit()
            logger.info(f"Added/updated client: {client_data['id']}")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error adding client {client_data.get('id')}: {str(e)}")
            return False
        finally:
            session.close()
    
    def add_document(self, document_data: Dict) -> Optional[int]:
        """Add document metadata."""

        # Normalize file_path to be relative
        if "file_path" in document_data:
            project_root = Path(__file__).parent.parent.parent.parent  # Adjust as needed
            try:
                document_data["file_path"] = str(Path(document_data["file_path"]).relative_to(project_root))
            except ValueError:
                # If already relative, leave as is
                document_data["file_path"] = str(Path(document_data["file_path"]))
        session = self.get_session()
        try:
            # Check if document exists
            existing_doc = session.query(Document).filter(
                Document.document_id == document_data["document_id"]
            ).first()
            
            if existing_doc:
                # Update existing document
                for key, value in document_data.items():
                    if hasattr(existing_doc, key) and key != "document_pk":
                        setattr(existing_doc, key, value)
                existing_doc.processed_date = datetime.utcnow()
                session.commit()
                return existing_doc.document_pk
            else:
                # Create new document
                document = Document(**document_data)
                session.add(document)
                session.commit()
                logger.info(f"Added document: {document_data['document_id']}")
                return document.document_pk
                
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error adding document {document_data.get('document_id')}: {str(e)}")
            return None
        finally:
            session.close()
    
    def add_document_chunks(self, chunks_data: List[Dict]) -> bool:
        """Add multiple document chunks."""
        session = self.get_session()
        try:
            for chunk_data in chunks_data:
                chunk = DocumentChunk(**chunk_data)
                session.add(chunk)
            
            session.commit()
            logger.info(f"Added {len(chunks_data)} document chunks")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error adding document chunks: {str(e)}")
            return False
        finally:
            session.close()
    
    def add_financial_metrics(self, metrics_data: List[Dict]) -> bool:
        """Add financial metrics."""
        session = self.get_session()
        try:
            for metric_data in metrics_data:
                metric = FinancialMetric(**metric_data)
                session.add(metric)
            
            session.commit()
            logger.info(f"Added {len(metrics_data)} financial metrics")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error adding financial metrics: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_client_documents(self, client_id: str) -> List[Dict]:
        """Get all documents for a client."""
        session = self.get_session()
        try:
            documents = session.query(Document).filter(Document.client_id == client_id).all()
            
            result = []
            for doc in documents:
                result.append({
                    "document_pk": doc.document_pk,
                    "document_id": doc.document_id,
                    "filing_type": doc.filing_type,
                    "filing_date": doc.filing_date,
                    "file_path": doc.file_path,
                    "total_chunks": doc.total_chunks,
                    "financial_density": doc.financial_density,
                    "has_revenue_data": doc.has_revenue_data,
                    "has_profit_data": doc.has_profit_data,
                    "has_balance_sheet": doc.has_balance_sheet,
                    "has_cash_flow": doc.has_cash_flow
                })
            
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting documents for client {client_id}: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_financial_metrics(
        self, 
        client_id: Optional[str] = None,
        metric_names: Optional[List[str]] = None,
        fiscal_year: Optional[int] = None
    ) -> pd.DataFrame:
        """Get financial metrics with optional filtering."""
        session = self.get_session()
        try:
            query = session.query(FinancialMetric)
            
            if client_id:
                query = query.filter(FinancialMetric.client_id == client_id)
            
            if metric_names:
                query = query.filter(FinancialMetric.metric_name.in_(metric_names))
            
            if fiscal_year:
                query = query.filter(FinancialMetric.fiscal_year == fiscal_year)
            
            metrics = query.all()
            
            # Convert to DataFrame
            data = []
            for metric in metrics:
                data.append({
                    "client_id": metric.client_id,
                    "metric_name": metric.metric_name,
                    "metric_value": metric.metric_value,
                    "metric_unit": metric.metric_unit,
                    "period_type": metric.period_type,
                    "period_end_date": metric.period_end_date,
                    "fiscal_year": metric.fiscal_year,
                    "fiscal_quarter": metric.fiscal_quarter,
                    "source_section": metric.source_section,
                    "extraction_confidence": metric.extraction_confidence
                })
            
            return pd.DataFrame(data)
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting financial metrics: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_client_statistics(self) -> pd.DataFrame:
        """Get statistics for all clients."""
        session = self.get_session()
        try:
            # Query with joins to get comprehensive stats
            from sqlalchemy import text
            query = text("""
                SELECT 
                    c.id as client_id,
                    c.company_name,
                    c.industry,
                    COUNT(DISTINCT d.document_pk) as document_count,
                    COUNT(DISTINCT dc.id) as chunk_count,
                    COUNT(DISTINCT fm.id) as metric_count,
                    AVG(d.financial_density) as avg_financial_density,
                    MAX(d.processed_date) as last_processed
                FROM clients c
                LEFT JOIN documents d ON c.id = d.client_id
                LEFT JOIN document_chunks dc ON d.document_pk = dc.document_pk
                LEFT JOIN financial_metrics fm ON c.id = fm.client_id
                GROUP BY c.id, c.company_name, c.industry
            """)
            
            result = session.execute(query)
            data = []
            for row in result:
                data.append({
                    'client_id': row[0],
                    'company_name': row[1],
                    'industry': row[2],
                    'document_count': row[3],
                    'chunk_count': row[4],
                    'metric_count': row[5],
                    'avg_financial_density': row[6],
                    'last_processed': row[7]
                })
            
            return pd.DataFrame(data)
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting client statistics: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def search_documents(
        self,
        client_id: Optional[str] = None,
        filing_types: Optional[List[str]] = None,
        has_financial_data: Optional[bool] = None,
        min_financial_density: Optional[float] = None
    ) -> List[Dict]:
        """Search documents with various filters."""
        session = self.get_session()
        try:
            query = session.query(Document)
            
            if client_id:
                query = query.filter(Document.client_id == client_id)
            
            if filing_types:
                query = query.filter(Document.filing_type.in_(filing_types))
            
            if has_financial_data:
                query = query.filter(
                    (Document.has_revenue_data == True) |
                    (Document.has_profit_data == True) |
                    (Document.has_balance_sheet == True) |
                    (Document.has_cash_flow == True)
                )
            
            if min_financial_density:
                query = query.filter(Document.financial_density >= min_financial_density)
            
            documents = query.all()
            
            result = []
            for doc in documents:
                result.append({
                    "document_id": doc.document_id,
                    "client_id": doc.client_id,
                    "filing_type": doc.filing_type,
                    "filing_date": doc.filing_date,
                    "file_path": doc.file_path,
                    "financial_density": doc.financial_density,
                    "total_chunks": doc.total_chunks
                })
            
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
        finally:
            session.close()
    
    def delete_client_data(self, client_id: str) -> bool:
        """Delete all data for a specific client."""
        session = self.get_session()
        try:
            # Delete in proper order due to foreign key constraints
            session.query(FinancialMetric).filter(FinancialMetric.client_id == client_id).delete()
            
            # Get document primary keys for chunk deletion
            doc_pks = [doc.document_pk for doc in session.query(Document.document_pk).filter(Document.client_id == client_id).all()]
            if doc_pks:
                session.query(DocumentChunk).filter(DocumentChunk.document_pk.in_(doc_pks)).delete()
            
            session.query(Document).filter(Document.client_id == client_id).delete()
            session.query(Client).filter(Client.id == client_id).delete()
            
            session.commit()
            logger.info(f"Deleted all data for client: {client_id}")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error deleting client data for {client_id}: {str(e)}")
            return False
        finally:
            session.close()


def main():
    """Example usage of SQL store."""
    sql_store = FinancialSQLStore()
    
    # Example of how client data would be added during SEC filing processing
    # This would normally happen automatically in sec_downloader.py when processing new filings
    client_data = {
        "id": "RDDT",
        "company_name": "Reddit, Inc.",
        "cik": "0001713445",
        "industry": "Technology",  # These could be fetched from SEC/financial data APIs
        "sector": "Social Media",
        "market_cap": None
    }
    sql_store.add_client(client_data)
    
    # Example document processing
    doc_data = {
        "document_id": "0001713445-25-000196",  # Actual SEC accession number
        "client_id": "RDDT",
        "filing_type": "10-Q",
        "filing_date": "2025-06-30",
        "file_path": "/path/to/document.txt",
        "file_size": 1000000,
        "download_date": datetime.utcnow(),
        "has_revenue_data": True,
        "has_profit_data": True,
        "financial_density": 15.5
    }
    doc_id = sql_store.add_document(doc_data)
    
    # Example metrics extraction
    metrics_data = [
        {
            "client_id": "RDDT",
            "document_pk": doc_id,
            "metric_name": "revenue",
            "metric_value": 500000000,
            "metric_unit": "USD",
            "period_type": "quarterly",
            "fiscal_year": 2025,
            "fiscal_quarter": 2,
            "source_section": "Consolidated Statements of Operations"
        }
    ]
    sql_store.add_financial_metrics(metrics_data)
    
    # Query and display results
    print("\nClient Statistics:")
    stats = sql_store.get_client_statistics()
    print(stats)
    
    print("\nFinancial Metrics for RDDT:")
    metrics = sql_store.get_financial_metrics(client_id="RDDT")
    print(metrics)


if __name__ == "__main__":
    main()