"""
Pytest configuration and shared fixtures for Financial Insight AI tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sqlite3
import json

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config.settings import get_settings
from knowledge_base.src.storage.sql_store import FinancialSQLStore
from knowledge_base.src.storage.vector_store import FinancialVectorStore


@pytest.fixture(scope="session")
def test_settings():
    """Provide test settings with temporary paths."""
    settings = get_settings()
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Override data paths for testing
        settings.data.raw_data_path = str(temp_path / "raw")
        settings.data.processed_data_path = str(temp_path / "processed")
        settings.data.output_path = str(temp_path / "outputs")
        settings.data.sqlite_db_path = str(temp_path / "test_financial_kb.db")
        settings.data.faiss_index_path = str(temp_path / "test_faiss_index")
        settings.data.chromadb_path = str(temp_path / "test_chromadb")
        
        # Create directories
        Path(settings.data.raw_data_path).mkdir(parents=True, exist_ok=True)
        Path(settings.data.processed_data_path).mkdir(parents=True, exist_ok=True)
        Path(settings.data.output_path).mkdir(parents=True, exist_ok=True)
        
        yield settings


@pytest.fixture
def temp_db(test_settings):
    """Provide a temporary SQLite database for testing."""
    db_path = Path(test_settings.data.sqlite_db_path)
    
    # Create fresh database
    if db_path.exists():
        db_path.unlink()
    
    # Initialize database
    sql_store = FinancialSQLStore()
    sql_store._create_tables()
    
    yield sql_store
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_sec_response():
    """Provide mock SEC API responses for testing."""
    return {
        "cik": "0000320193",
        "entityType": "operating",
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "fiscalYearEnd": "0930",
        "addresses": {
            "mailing": {
                "street1": "ONE APPLE PARK WAY",
                "street2": "",
                "city": "CUPERTINO",
                "stateOrCountry": "CA",
                "zipCode": "95014",
                "stateOrCountryDescription": "CA"
            }
        }
    }


@pytest.fixture
def sample_sec_filing():
    """Provide sample SEC filing data for testing."""
    return {
        "accession_number": "0000320193-22-000108",
        "type": "10-K",
        "period_of_report": "2022-09-24",
        "file_path": "/path/to/filing.txt",
        "file_size": 10332356,
        "downloaded_at": "2025-08-03T12:25:01.997157",
        "has_revenue_data": True,
        "has_profit_data": True,
        "has_balance_sheet": True,
        "has_cash_flow": True,
        "document_id": "0000320193-22-000108"
    }


@pytest.fixture
def sample_financial_metrics():
    """Provide sample financial metrics for testing."""
    return [
        {
            "metric_name": "revenue",
            "value": 394328000000,
            "currency": "USD",
            "period": "2022",
            "filing_date": "2022-09-24",
            "source_section": "Consolidated Statements of Operations",
            "confidence_score": 0.95
        },
        {
            "metric_name": "net_income",
            "value": 96995000000,
            "currency": "USD", 
            "period": "2022",
            "filing_date": "2022-09-24",
            "source_section": "Consolidated Statements of Operations",
            "confidence_score": 0.92
        }
    ]


@pytest.fixture
def mock_downloader():
    """Provide a mock SEC downloader for testing."""
    with patch('knowledge_base.src.ingestion.sec_downloader.SECDownloader') as mock:
        downloader = mock.return_value
        downloader.download_company_filings.return_value = [
            {
                "accession_number": "0000320193-22-000108",
                "type": "10-K",
                "period_of_report": "2022-09-24",
                "file_path": "/test/path/filing.txt",
                "file_size": 10332356,
                "downloaded_at": "2025-08-03T12:25:01.997157",
                "has_revenue_data": True,
                "has_profit_data": True,
                "has_balance_sheet": True,
                "has_cash_flow": True,
                "document_id": "0000320193-22-000108"
            }
        ]
        yield downloader


@pytest.fixture
def sample_document_chunk():
    """Provide sample document chunk for testing."""
    return {
        "content": "Apple Inc. reported revenue of $394.3 billion for fiscal year 2022.",
        "metadata": {
            "source": "SEC Edgar",
            "filing_type": "10-K",
            "ticker": "AAPL",
            "filing_date": "2022-09-24",
            "section": "Management Discussion and Analysis",
            "chunk_id": "chunk_001"
        },
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding
    } 