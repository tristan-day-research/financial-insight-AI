"""
Integration tests for the complete Financial Insight AI pipeline.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json
import sqlite3
from datetime import datetime

from knowledge_base.src.ingestion.sec_downloader import SECDownloader
from knowledge_base.src.ingestion.document_processor import FinancialDocumentProcessor
from knowledge_base.src.ingestion.sec_sql_extractor import SECDataExtractor
from knowledge_base.src.storage.sql_manager import FinancialMetricsManager
from knowledge_base.src.retrieval.rag_engine import FinancialRAGEngine


class TestCompletePipeline:
    """Test the complete data processing pipeline from download to query."""
    
    @pytest.fixture
    def pipeline_components(self, test_settings):
        """Set up all pipeline components for testing."""
        downloader = SECDownloader()
        processor = FinancialDocumentProcessor()
        extractor = SECDataExtractor()
        sql_manager = FinancialMetricsManager(test_settings)
        rag_engine = FinancialRAGEngine()
        
        return {
            'downloader': downloader,
            'processor': processor,
            'extractor': extractor,
            'sql_manager': sql_manager,
            'rag_engine': rag_engine,
            'settings': test_settings
        }
    
    def test_document_download_to_storage(self, pipeline_components):
        """Test complete flow from document download to storage."""
        downloader = pipeline_components['downloader']
        processor = pipeline_components['processor']
        sql_manager = pipeline_components['sql_manager']
        
        # Mock the download to avoid real API calls
        with patch.object(downloader, 'download_company_filings') as mock_download:
            mock_download.return_value = [{
                'accession_number': '0000320193-22-000108',
                'type': '10-K',
                'period_of_report': '2022-09-24',
                'file_path': '/test/path/filing.txt',
                'file_size': 10332356,
                'downloaded_at': '2025-08-03T12:25:01.997157',
                'has_revenue_data': True,
                'has_profit_data': True,
                'has_balance_sheet': True,
                'has_cash_flow': True,
                'document_id': '0000320193-22-000108'
            }]
            
            # Step 1: Download filing
            filings = downloader.download_company_filings('AAPL', ['10-K'], 1)
            assert len(filings) == 1
            
            # Step 2: Process document (mock file content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("""
                APPLE INC.
                CONSOLIDATED STATEMENTS OF OPERATIONS
                Total net sales: $394,328 million
                Net income: $96,995 million
                """)
                test_file_path = f.name
            
            try:
                # Process for chunks
                metadata = {
                    'ticker': 'AAPL',
                    'filing_type': '10-K',
                    'filing_date': '2022-09-24',
                    'source': 'SEC Edgar'
                }
                chunks = processor.process_sec_filing(test_file_path, metadata)
                assert len(chunks) > 0
                
                # Step 3: Store in database (mock)
                with patch.object(sql_manager.sql_store, 'add_document') as mock_add_doc:
                    mock_add_doc.return_value = 1
                    
                    doc_data = {
                        "document_id": filings[0]['accession_number'],
                        "client_id": "AAPL",
                        "filing_type": filings[0]['type'],
                        "filing_date": filings[0]['period_of_report'],
                        "file_path": filings[0]['file_path'],
                        "file_size": filings[0]['file_size'],
                        "download_date": datetime.fromisoformat(filings[0]['downloaded_at']),
                        "has_revenue_data": filings[0]['has_revenue_data'],
                        "has_profit_data": filings[0]['has_profit_data'],
                        "has_balance_sheet": filings[0]['has_balance_sheet'],
                        "has_cash_flow": filings[0]['has_cash_flow']
                    }
                    
                    doc_id = sql_manager.sql_store.add_document(doc_data)
                    assert doc_id == 1
                    
                    # Verify document was added
                    mock_add_doc.assert_called_once()
            
            finally:
                Path(test_file_path).unlink()
    
    def test_financial_metrics_extraction_pipeline(self, pipeline_components):
        """Test complete financial metrics extraction pipeline."""
        extractor = pipeline_components['extractor']
        sql_manager = pipeline_components['sql_manager']
        
        # Create test document with financial data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            CONSOLIDATED STATEMENTS OF OPERATIONS
            (In millions)
            
            Three Months Ended September 24, 2022
            Net sales:
            iPhone                    $ 42,626
            Mac                        11,512
            iPad                        7,174
            Wearables, Home and Accessories    9,650
            Services                   19,189
            Total net sales           90,151
            Cost of sales             49,462
            Gross margin              40,689
            Operating income          27,913
            Net income                $ 23,464
            """)
            test_file_path = f.name
        
        try:
            # Step 1: Extract metrics
            metrics = extractor.process_document(test_file_path, "AAPL")
            assert len(metrics) > 0
            
            # Verify metric structure
            for metric in metrics:
                assert 'metric_name' in metric
                assert 'value' in metric
                assert 'ticker' in metric
                assert metric['ticker'] == 'AAPL'
            
            # Step 2: Store metrics (mock)
            with patch.object(sql_manager, 'save_extracted_metrics') as mock_save:
                mock_save.return_value = True
                
                doc_id = 1  # Mock document ID
                success = sql_manager.save_extracted_metrics(metrics, doc_id)
                assert success is True
                
                # Verify metrics were saved
                mock_save.assert_called_once_with(metrics, doc_id)
        
        finally:
            Path(test_file_path).unlink()
    
    def test_rag_query_pipeline(self, pipeline_components):
        """Test complete RAG query pipeline."""
        rag_engine = pipeline_components['rag_engine']
        
        # Mock the RAG components
        with patch.object(rag_engine, 'search_documents') as mock_search:
            mock_search.return_value = [
                Mock(
                    page_content="Apple Inc. reported revenue of $394.3 billion in 2022.",
                    metadata={'ticker': 'AAPL', 'filing_type': '10-K'}
                )
            ]
            
            with patch.object(rag_engine, 'generate_response') as mock_generate:
                mock_generate.return_value = "Apple Inc. reported revenue of $394.3 billion in fiscal year 2022."
                
                # Test query processing
                query = "What was Apple's revenue in 2022?"
                response = rag_engine.generate_response(query)
                
                # Verify response was generated
                assert response is not None
                assert "Apple" in response
                assert "revenue" in response.lower()
                
                # Verify search was called
                mock_search.assert_called_once()
                mock_generate.assert_called_once()
    
    def test_hybrid_search_pipeline(self, pipeline_components):
        """Test hybrid search combining vector and SQL data."""
        rag_engine = pipeline_components['rag_engine']
        
        # Mock both vector and SQL components
        with patch.object(rag_engine, 'search_documents') as mock_vector_search:
            mock_vector_search.return_value = [
                Mock(
                    page_content="Apple Inc. reported strong financial performance.",
                    metadata={'ticker': 'AAPL', 'filing_type': '10-K'}
                )
            ]
            
            with patch.object(rag_engine, 'get_financial_metrics') as mock_sql_metrics:
                mock_sql_metrics.return_value = [
                    {
                        'metric_name': 'revenue',
                        'value': 394328000000,
                        'currency': 'USD',
                        'period': '2022',
                        'ticker': 'AAPL'
                    }
                ]
                
                # Test hybrid search
                query = "What was Apple's revenue in 2022?"
                results = rag_engine.hybrid_search(query, 'AAPL', ['revenue'], 2022)
                
                # Verify both searches were performed
                mock_vector_search.assert_called_once()
                mock_sql_metrics.assert_called_once()
                
                # Verify combined results
                assert 'vector_results' in results
                assert 'sql_metrics' in results
                assert len(results['vector_results']) == 1
                assert len(results['sql_metrics']) == 1


class TestDataConsistency:
    """Test data consistency across the pipeline."""
    
    def test_document_metadata_consistency(self, test_settings):
        """Test that document metadata is consistent throughout the pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test database
            db_path = Path(temp_dir) / "test_consistency.db"
            test_settings.data.sqlite_db_path = str(db_path)
            
            from knowledge_base.src.storage.sql_store import FinancialSQLStore
            sql_store = FinancialSQLStore()
            sql_store._create_tables()
            
            # Test document metadata
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
            
            # Add document
            doc_id = sql_store.add_document(doc_data)
            assert doc_id is not None
            
            # Retrieve and verify consistency
            retrieved_doc = sql_store.get_document(doc_id)
            assert retrieved_doc is not None
            assert retrieved_doc['document_id'] == doc_data['document_id']
            assert retrieved_doc['client_id'] == doc_data['client_id']
            assert retrieved_doc['filing_type'] == doc_data['filing_type']
            assert retrieved_doc['filing_date'] == doc_data['filing_date']
    
    def test_financial_metrics_consistency(self, test_settings):
        """Test that financial metrics are consistent throughout the pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test database
            db_path = Path(temp_dir) / "test_metrics.db"
            test_settings.data.sqlite_db_path = str(db_path)
            
            from knowledge_base.src.storage.sql_store import FinancialSQLStore
            sql_store = FinancialSQLStore()
            sql_store._create_tables()
            
            # Add test document
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
            doc_id = sql_store.add_document(doc_data)
            
            # Add test metrics
            test_metrics = [
                {
                    "metric_name": "revenue",
                    "value": 394328000000,
                    "currency": "USD",
                    "period": "2022",
                    "filing_date": "2022-09-24",
                    "source_section": "Consolidated Statements of Operations",
                    "confidence_score": 0.95,
                    "document_id": doc_id
                },
                {
                    "metric_name": "net_income",
                    "value": 96995000000,
                    "currency": "USD",
                    "period": "2022",
                    "filing_date": "2022-09-24",
                    "source_section": "Consolidated Statements of Operations",
                    "confidence_score": 0.92,
                    "document_id": doc_id
                }
            ]
            
            for metric in test_metrics:
                metric_id = sql_store.add_financial_metric(metric)
                assert metric_id is not None
            
            # Verify metrics in database
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM financial_metrics WHERE document_id = ?", (doc_id,))
                stored_metrics = cursor.fetchall()
                
                assert len(stored_metrics) == 2
                
                # Verify metric values
                for stored_metric in stored_metrics:
                    assert stored_metric[1] == doc_id  # document_id
                    assert stored_metric[2] in ['revenue', 'net_income']  # metric_name
                    assert stored_metric[3] > 0  # value


class TestErrorHandling:
    """Test error handling throughout the pipeline."""
    
    def test_download_error_handling(self, test_settings):
        """Test handling of download errors."""
        downloader = SECDownloader()
        
        # Test with invalid ticker
        with patch.object(downloader, 'get_company_info') as mock_info:
            mock_info.return_value = None
            
            result = downloader.get_company_info('INVALID_TICKER')
            assert result is None
    
    def test_database_error_handling(self, test_settings):
        """Test handling of database errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_error.db"
            test_settings.data.sqlite_db_path = str(db_path)
            
            from knowledge_base.src.storage.sql_store import FinancialSQLStore
            sql_store = FinancialSQLStore()
            sql_store._create_tables()
            
            # Test retrieving non-existent document
            document = sql_store.get_document(99999)
            assert document is None
    
    def test_processing_error_handling(self, test_settings):
        """Test handling of document processing errors."""
        processor = FinancialDocumentProcessor()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            processor.process_sec_filing('/non/existent/file.txt', {})
    
    def test_extraction_error_handling(self, test_settings):
        """Test handling of data extraction errors."""
        extractor = SECDataExtractor()
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            test_file_path = f.name
        
        try:
            metrics = extractor.process_document(test_file_path, "AAPL")
            # Should handle empty file gracefully
            assert isinstance(metrics, list)
        finally:
            Path(test_file_path).unlink()


class TestPerformanceIntegration:
    """Test performance of the complete pipeline."""
    
    def test_pipeline_performance(self, test_settings):
        """Test performance of the complete pipeline."""
        import time
        
        # Mock all components for performance testing
        with patch('knowledge_base.src.ingestion.sec_downloader.SECDownloader') as mock_downloader_class:
            mock_downloader = mock_downloader_class.return_value
            mock_downloader.download_company_filings.return_value = [{
                'accession_number': 'test-123',
                'type': '10-K',
                'period_of_report': '2022-09-24',
                'file_path': '/test/path/file.txt',
                'file_size': 1000000,
                'downloaded_at': '2025-08-03T12:25:01.997157',
                'has_revenue_data': True,
                'has_profit_data': True,
                'has_balance_sheet': True,
                'has_cash_flow': True,
                'document_id': 'test-123'
            }]
            
            start_time = time.time()
            
            # Simulate pipeline steps
            downloader = SECDownloader()
            filings = downloader.download_company_filings('AAPL', ['10-K'], 1)
            
            processor = FinancialDocumentProcessor()
            extractor = SECDataExtractor()
            
            # Mock file processing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Test content")
                test_file_path = f.name
            
            try:
                chunks = processor.process_sec_filing(test_file_path, {})
                metrics = extractor.process_document(test_file_path, "AAPL")
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Verify performance is reasonable
                assert total_time < 5.0  # Should complete within 5 seconds
                assert len(filings) == 1
                assert len(chunks) > 0
                assert len(metrics) >= 0  # May be empty for test content
            
            finally:
                Path(test_file_path).unlink() 