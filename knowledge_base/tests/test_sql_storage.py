"""
Unit tests for SQL storage functionality.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import sqlite3
from datetime import datetime

from knowledge_base.src.storage.sql_store import FinancialSQLStore
from knowledge_base.src.storage.sql_manager import FinancialMetricsManager


class TestSQLStore:
    """Test cases for SQLStore class."""
    
    def test_initialization(self, test_settings):
        """Test FinancialSQLStore initialization."""
        sql_store = FinancialSQLStore()
        assert sql_store is not None
        assert hasattr(sql_store, 'engine')
        assert hasattr(sql_store, 'get_session')
    
    def test_database_creation(self, temp_db):
        """Test database creation and table initialization."""
        # Verify tables were created
        db_path = Path(temp_db.settings.database.sqlite_db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Check that all expected tables exist
            expected_tables = ['clients', 'documents', 'document_chunks', 'financial_metrics']
            for table in expected_tables:
                assert table in tables
    
    def test_add_document(self, temp_db, sample_sec_filing):
        """Test adding a document to the database."""
        # Prepare document data
        doc_data = {
            "document_id": sample_sec_filing['accession_number'],
            "client_id": "TEST_CLIENT",
            "filing_type": sample_sec_filing['type'],
            "filing_date": sample_sec_filing['period_of_report'],
            "file_path": sample_sec_filing['file_path'],
            "file_size": sample_sec_filing['file_size'],
            "download_date": datetime.fromisoformat(sample_sec_filing['downloaded_at']),
            "has_revenue_data": sample_sec_filing['has_revenue_data'],
            "has_profit_data": sample_sec_filing['has_profit_data'],
            "has_balance_sheet": sample_sec_filing['has_balance_sheet'],
            "has_cash_flow": sample_sec_filing['has_cash_flow']
        }
        
        # Add document
        doc_id = temp_db.add_document(doc_data)
        
        # Verify document was added
        assert doc_id is not None
        assert isinstance(doc_id, int)
        
        # Verify document exists in database
        db_path = Path(temp_db.settings.database.sqlite_db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            result = cursor.fetchone()
            assert result is not None
    
    def test_add_client(self, temp_db):
        """Test adding a client to the database."""
        client_data = {
            "company_name": "Apple Inc.",
            "cik": "0000320193",
            "industry": "Technology",
            "sector": "Consumer Electronics",
            "market_cap": 2500000000000
        }
        
        # Add client
        client_id = temp_db.add_client(client_data)
        
        # Verify client was added
        assert client_id is not None
        assert isinstance(client_id, int)
        
        # Verify client exists in database
        db_path = Path(temp_db.settings.database.sqlite_db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clients WHERE id = ?", (client_id,))
            result = cursor.fetchone()
            assert result is not None
            assert result[1] == "Apple Inc."  # company_name
    
    def test_add_financial_metrics(self, temp_db, sample_financial_metrics):
        """Test adding financial metrics to the database."""
        # First add a document
        doc_data = {
            "document_id": "test-doc-123",
            "client_id": "TEST_CLIENT",
            "filing_type": "10-K",
            "filing_date": "2022-09-24",
            "file_path": "/test/path/file.txt",
            "file_size": 1000000,
            "download_date": datetime.now(),
            "has_revenue_data": True,
            "has_profit_data": True,
            "has_balance_sheet": True,
            "has_cash_flow": True
        }
        doc_id = temp_db.add_document(doc_data)
        
        # Add metrics
        for metric in sample_financial_metrics:
            metric['document_id'] = doc_id
            metric_id = temp_db.add_financial_metric(metric)
            assert metric_id is not None
        
        # Verify metrics were added
        db_path = Path(temp_db.settings.database.sqlite_db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM financial_metrics WHERE document_id = ?", (doc_id,))
            count = cursor.fetchone()[0]
            assert count == len(sample_financial_metrics)
    
    def test_get_document(self, temp_db, sample_sec_filing):
        """Test retrieving a document from the database."""
        # Add a document first
        doc_data = {
            "document_id": sample_sec_filing['accession_number'],
            "client_id": "TEST_CLIENT",
            "filing_type": sample_sec_filing['type'],
            "filing_date": sample_sec_filing['period_of_report'],
            "file_path": sample_sec_filing['file_path'],
            "file_size": sample_sec_filing['file_size'],
            "download_date": datetime.fromisoformat(sample_sec_filing['downloaded_at']),
            "has_revenue_data": sample_sec_filing['has_revenue_data'],
            "has_profit_data": sample_sec_filing['has_profit_data'],
            "has_balance_sheet": sample_sec_filing['has_balance_sheet'],
            "has_cash_flow": sample_sec_filing['has_cash_flow']
        }
        doc_id = temp_db.add_document(doc_data)
        
        # Retrieve the document
        document = temp_db.get_document(doc_id)
        
        # Verify document data
        assert document is not None
        assert document['document_id'] == sample_sec_filing['accession_number']
        assert document['filing_type'] == sample_sec_filing['type']
    
    def test_get_document_not_found(self, temp_db):
        """Test retrieving a non-existent document."""
        document = temp_db.get_document(99999)
        assert document is None
    
    def test_update_document(self, temp_db, sample_sec_filing):
        """Test updating a document in the database."""
        # Add a document first
        doc_data = {
            "document_id": sample_sec_filing['accession_number'],
            "client_id": "TEST_CLIENT",
            "filing_type": sample_sec_filing['type'],
            "filing_date": sample_sec_filing['period_of_report'],
            "file_path": sample_sec_filing['file_path'],
            "file_size": sample_sec_filing['file_size'],
            "download_date": datetime.fromisoformat(sample_sec_filing['downloaded_at']),
            "has_revenue_data": sample_sec_filing['has_revenue_data'],
            "has_profit_data": sample_sec_filing['has_profit_data'],
            "has_balance_sheet": sample_sec_filing['has_balance_sheet'],
            "has_cash_flow": sample_sec_filing['has_cash_flow']
        }
        doc_id = temp_db.add_document(doc_data)
        
        # Update the document
        update_data = {
            "processed_date": datetime.now(),
            "total_chunks": 150,
            "financial_density": 0.85
        }
        success = temp_db.update_document(doc_id, update_data)
        
        # Verify update was successful
        assert success is True
        
        # Verify the update
        document = temp_db.get_document(doc_id)
        assert document['total_chunks'] == 150
        assert document['financial_density'] == 0.85


class TestFinancialMetricsManager:
    """Test cases for FinancialMetricsManager class."""
    
    def test_initialization(self, test_settings):
        """Test FinancialMetricsManager initialization."""
        manager = FinancialMetricsManager(test_settings)
        assert manager is not None
        assert hasattr(manager, 'sql_store')
    
    def test_save_extracted_metrics(self, temp_db, sample_financial_metrics):
        """Test saving extracted financial metrics."""
        manager = FinancialMetricsManager(temp_db.settings)
        manager.sql_store = temp_db
        
        # First add a document
        doc_data = {
            "document_id": "test-doc-123",
            "client_id": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2022-09-24",
            "file_path": "/test/path/file.txt",
            "file_size": 1000000,
            "download_date": datetime.now(),
            "has_revenue_data": True,
            "has_profit_data": True,
            "has_balance_sheet": True,
            "has_cash_flow": True
        }
        doc_id = temp_db.add_document(doc_data)
        
        # Save metrics
        success = manager.save_extracted_metrics(sample_financial_metrics, doc_id)
        
        # Verify metrics were saved
        assert success is True
        
        # Verify metrics in database
        with sqlite3.connect(temp_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM financial_metrics WHERE document_id = ?", (doc_id,))
            count = cursor.fetchone()[0]
            assert count == len(sample_financial_metrics)
    
    def test_validate_client_metrics(self, temp_db, sample_financial_metrics):
        """Test validating client metrics."""
        manager = FinancialMetricsManager(temp_db.settings)
        manager.sql_store = temp_db
        
        # Add test data
        doc_data = {
            "document_id": "test-doc-123",
            "client_id": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2022-09-24",
            "file_path": "/test/path/file.txt",
            "file_size": 1000000,
            "download_date": datetime.now(),
            "has_revenue_data": True,
            "has_profit_data": True,
            "has_balance_sheet": True,
            "has_cash_flow": True
        }
        doc_id = temp_db.add_document(doc_data)
        manager.save_extracted_metrics(sample_financial_metrics, doc_id)
        
        # Validate metrics
        validation = manager.validate_client_metrics("AAPL", 2022)
        
        # Verify validation results
        assert validation is not None
        assert 'total_metrics' in validation
        assert 'valid_metrics' in validation
        assert 'missing_metrics' in validation
    
    def test_get_comparative_metrics(self, temp_db, sample_financial_metrics):
        """Test getting comparative metrics across companies."""
        manager = FinancialMetricsManager(temp_db.settings)
        manager.sql_store = temp_db
        
        # Add test data for multiple companies
        companies = ["AAPL", "MSFT", "GOOGL"]
        for i, company in enumerate(companies):
            doc_data = {
                "document_id": f"test-doc-{i}",
                "client_id": company,
                "filing_type": "10-K",
                "filing_date": "2022-09-24",
                "file_path": f"/test/path/{company}.txt",
                "file_size": 1000000,
                "download_date": datetime.now(),
                "has_revenue_data": True,
                "has_profit_data": True,
                "has_balance_sheet": True,
                "has_cash_flow": True
            }
            doc_id = temp_db.add_document(doc_data)
            
            # Modify metrics for each company
            company_metrics = []
            for metric in sample_financial_metrics:
                company_metric = metric.copy()
                company_metric['value'] = metric['value'] * (i + 1)  # Different values per company
                company_metrics.append(company_metric)
            
            manager.save_extracted_metrics(company_metrics, doc_id)
        
        # Get comparative metrics
        comparative = manager.get_comparative_metrics(
            companies,
            ["revenue", "net_income"],
            2022
        )
        
        # Verify comparative results
        assert comparative is not None
        assert len(comparative) == len(companies)
        for company_data in comparative:
            assert 'client_id' in company_data
            assert 'metrics' in company_data


class TestSQLStorageIntegration:
    """Integration tests for SQL storage with real database operations."""
    
    @pytest.mark.integration
    def test_full_document_lifecycle(self, test_settings):
        """Test complete document lifecycle in SQL storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary database
            db_path = Path(temp_dir) / "test_integration.db"
            test_settings.data.sqlite_db_path = str(db_path)
            
            sql_store = SQLStore(test_settings)
            sql_store.initialize_database()
            
            # Add client
            client_data = {
                "company_name": "Apple Inc.",
                "cik": "0000320193",
                "industry": "Technology",
                "sector": "Consumer Electronics",
                "market_cap": 2500000000000
            }
            client_id = sql_store.add_client(client_data)
            assert client_id is not None
            
            # Add document
            doc_data = {
                "document_id": "0000320193-22-000108",
                "client_id": "AAPL",
                "filing_type": "10-K",
                "filing_date": "2022-09-24",
                "file_path": "/test/path/filing.txt",
                "file_size": 10332356,
                "download_date": datetime.now(),
                "has_revenue_data": True,
                "has_profit_data": True,
                "has_balance_sheet": True,
                "has_cash_flow": True
            }
            doc_id = sql_store.add_document(doc_data)
            assert doc_id is not None
            
            # Add financial metrics
            metrics = [
                {
                    "metric_name": "revenue",
                    "value": 394328000000,
                    "currency": "USD",
                    "period": "2022",
                    "filing_date": "2022-09-24",
                    "source_section": "Consolidated Statements of Operations",
                    "confidence_score": 0.95,
                    "document_id": doc_id
                }
            ]
            
            for metric in metrics:
                metric_id = sql_store.add_financial_metric(metric)
                assert metric_id is not None
            
            # Verify complete data integrity
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check client
                cursor.execute("SELECT * FROM clients WHERE id = ?", (client_id,))
                client = cursor.fetchone()
                assert client is not None
                assert client[1] == "Apple Inc."
                
                # Check document
                cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                document = cursor.fetchone()
                assert document is not None
                assert document[1] == "0000320193-22-000108"
                
                # Check metrics
                cursor.execute("SELECT COUNT(*) FROM financial_metrics WHERE document_id = ?", (doc_id,))
                metric_count = cursor.fetchone()[0]
                assert metric_count == 1 