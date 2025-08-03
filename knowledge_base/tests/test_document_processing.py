"""
Unit tests for document processing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from knowledge_base.src.ingestion.document_processor import FinancialDocumentProcessor
from knowledge_base.src.ingestion.sec_sql_extractor import SECDataExtractor


class TestFinancialDocumentProcessor:
    """Test cases for FinancialDocumentProcessor class."""
    
    @pytest.fixture
    def processor(self, test_settings):
        """Create a test instance of FinancialDocumentProcessor."""
        return FinancialDocumentProcessor()
    
    def test_initialization(self, processor):
        """Test FinancialDocumentProcessor initialization."""
        assert processor is not None
        assert hasattr(processor, 'chunk_size')
        assert hasattr(processor, 'chunk_overlap')
    
    def test_process_sec_filing(self, processor):
        """Test processing SEC filing documents."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = """
            APPLE INC.
            CONSOLIDATED STATEMENTS OF OPERATIONS
            (In millions, except number of shares which are reflected in thousands and per share amounts)
            
            Three Months Ended
            September 24, 2022    September 25, 2021
            Net sales:
            iPhone                    $ 42,626        $ 38,868
            Mac                        11,512          9,178
            iPad                        7,174          8,830
            Wearables, Home and Accessories    9,650          8,791
            Services                   19,189          18,277
            Total net sales           90,151          83,944
            Cost of sales             49,462          48,198
            Gross margin              40,689          35,746
            Operating expenses:
            Research and development   6,667           5,615
            Selling, general and administrative    6,109           5,614
            Total operating expenses  12,776          11,229
            Operating income          27,913          24,517
            Other income/(expense), net     (25)            (4)
            Income before provision for income taxes    27,888          24,513
            Provision for income taxes    4,424           3,254
            Net income                $ 23,464        $ 21,259
            """
            f.write(test_content)
            test_file_path = f.name
        
        try:
            # Test processing
            metadata = {
                'ticker': 'AAPL',
                'filing_type': '10-K',
                'filing_date': '2022-09-24',
                'source': 'SEC Edgar'
            }
            
            chunks = processor.process_sec_filing(test_file_path, metadata)
            
            # Verify chunks were created
            assert len(chunks) > 0
            
            # Verify chunk structure
            for chunk in chunks:
                assert hasattr(chunk, 'page_content')
                assert hasattr(chunk, 'metadata')
                assert chunk.metadata['ticker'] == 'AAPL'
                assert chunk.metadata['filing_type'] == '10-K'
                assert chunk.metadata['source'] == 'SEC Edgar'
        
        finally:
            # Cleanup
            Path(test_file_path).unlink()
    
    def test_save_processed_chunks(self, processor):
        """Test saving processed chunks to file."""
        # Create test chunks
        test_chunks = [
            Mock(
                page_content="Apple Inc. reported revenue of $394.3 billion.",
                metadata={
                    'ticker': 'AAPL',
                    'filing_type': '10-K',
                    'section': 'Management Discussion'
                }
            ),
            Mock(
                page_content="Net income was $96.9 billion for fiscal year 2022.",
                metadata={
                    'ticker': 'AAPL',
                    'filing_type': '10-K',
                    'section': 'Financial Statements'
                }
            )
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_chunks.json"
            
            # Save chunks
            processor.save_processed_chunks(test_chunks, str(output_file))
            
            # Verify file was created
            assert output_file.exists()
            
            # Verify file content
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
                assert len(saved_data) == 2
                assert saved_data[0]['content'] == "Apple Inc. reported revenue of $394.3 billion."
                assert saved_data[0]['metadata']['ticker'] == 'AAPL'
    
    def test_chunk_document(self, processor):
        """Test document chunking functionality."""
        # Test content
        content = """
        This is a test document with multiple paragraphs.
        
        It contains financial information about Apple Inc.
        
        Revenue for the quarter was $90.2 billion.
        
        Net income was $23.5 billion.
        """
        
        chunks = processor._chunk_document(content, {'ticker': 'AAPL'})
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify chunk content
        for chunk in chunks:
            assert len(chunk.page_content) > 0
            assert chunk.metadata['ticker'] == 'AAPL'
    
    def test_extract_financial_sections(self, processor):
        """Test extraction of financial sections from documents."""
        content = """
        CONSOLIDATED STATEMENTS OF OPERATIONS
        Revenue: $394,328 million
        
        BALANCE SHEET
        Total assets: $352,755 million
        
        CASH FLOWS
        Operating cash flow: $122,151 million
        
        MANAGEMENT DISCUSSION
        Our strategy focuses on innovation.
        """
        
        sections = processor._extract_financial_sections(content)
        
        # Verify sections were extracted
        assert 'CONSOLIDATED STATEMENTS OF OPERATIONS' in sections
        assert 'BALANCE SHEET' in sections
        assert 'CASH FLOWS' in sections
        assert 'MANAGEMENT DISCUSSION' in sections
    
    def test_clean_text(self, processor):
        """Test text cleaning functionality."""
        dirty_text = """
        \n\n\n  Apple Inc.  \n\n  reported  \n\n  revenue  \n\n  of  \n\n  $394.3  \n\n  billion.  \n\n\n
        """
        
        clean_text = processor._clean_text(dirty_text)
        
        # Verify text was cleaned
        assert clean_text.strip() == "Apple Inc. reported revenue of $394.3 billion."
        assert '\n\n\n' not in clean_text
        assert '  ' not in clean_text


class TestSECDataExtractor:
    """Test cases for SECDataExtractor class."""
    
    @pytest.fixture
    def extractor(self, test_settings):
        """Create a test instance of SECDataExtractor."""
        return SECDataExtractor()
    
    def test_initialization(self, extractor):
        """Test SECDataExtractor initialization."""
        assert extractor is not None
        assert hasattr(extractor, 'patterns')
        assert hasattr(extractor, 'confidence_threshold')
    
    def test_process_document(self, extractor):
        """Test processing a document for financial data extraction."""
        # Create test document content
        test_content = """
        CONSOLIDATED STATEMENTS OF OPERATIONS
        (In millions, except per share amounts)
        
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
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file_path = f.name
        
        try:
            # Process document
            metrics = extractor.process_document(test_file_path, "AAPL")
            
            # Verify metrics were extracted
            assert len(metrics) > 0
            
            # Verify metric structure
            for metric in metrics:
                assert 'metric_name' in metric
                assert 'value' in metric
                assert 'currency' in metric
                assert 'period' in metric
                assert 'confidence_score' in metric
                assert metric['ticker'] == 'AAPL'
        
        finally:
            Path(test_file_path).unlink()
    
    def test_extract_revenue(self, extractor):
        """Test revenue extraction from text."""
        text = """
        CONSOLIDATED STATEMENTS OF OPERATIONS
        Total net sales           90,151
        Revenue for the quarter was $90.2 billion.
        Annual revenue: $394,328 million
        """
        
        revenue_metrics = extractor._extract_revenue(text, "AAPL", "2022")
        
        # Verify revenue was extracted
        assert len(revenue_metrics) > 0
        
        for metric in revenue_metrics:
            assert metric['metric_name'] == 'revenue'
            assert metric['ticker'] == 'AAPL'
            assert metric['period'] == '2022'
            assert metric['value'] > 0
    
    def test_extract_net_income(self, extractor):
        """Test net income extraction from text."""
        text = """
        CONSOLIDATED STATEMENTS OF OPERATIONS
        Net income                $ 23,464
        Net income for the year was $96.9 billion.
        """
        
        income_metrics = extractor._extract_net_income(text, "AAPL", "2022")
        
        # Verify net income was extracted
        assert len(income_metrics) > 0
        
        for metric in income_metrics:
            assert metric['metric_name'] == 'net_income'
            assert metric['ticker'] == 'AAPL'
            assert metric['period'] == '2022'
            assert metric['value'] > 0
    
    def test_extract_balance_sheet_metrics(self, extractor):
        """Test balance sheet metrics extraction."""
        text = """
        CONSOLIDATED BALANCE SHEETS
        (In millions)
        
        September 24, 2022
        ASSETS
        Current assets:
        Cash and cash equivalents    $ 48,844
        Marketable securities        24,718
        Accounts receivable, net      28,184
        Inventories                   4,646
        Total current assets         106,392
        
        LIABILITIES
        Current liabilities:
        Accounts payable              $ 64,115
        Accrued expenses              47,236
        Total current liabilities    111,351
        """
        
        balance_metrics = extractor._extract_balance_sheet_metrics(text, "AAPL", "2022")
        
        # Verify balance sheet metrics were extracted
        assert len(balance_metrics) > 0
        
        metric_names = [m['metric_name'] for m in balance_metrics]
        assert 'total_assets' in metric_names or 'current_assets' in metric_names
        assert 'total_liabilities' in metric_names or 'current_liabilities' in metric_names
    
    def test_extract_cash_flow_metrics(self, extractor):
        """Test cash flow metrics extraction."""
        text = """
        CONSOLIDATED STATEMENTS OF CASH FLOWS
        (In millions)
        
        Year Ended September 24, 2022
        Cash generated by operating activities    $ 122,151
        Cash used in investing activities         (22,107)
        Cash used in financing activities         (110,544)
        """
        
        cash_flow_metrics = extractor._extract_cash_flow_metrics(text, "AAPL", "2022")
        
        # Verify cash flow metrics were extracted
        assert len(cash_flow_metrics) > 0
        
        for metric in cash_flow_metrics:
            assert 'cash_flow' in metric['metric_name'].lower()
            assert metric['ticker'] == 'AAPL'
            assert metric['period'] == '2022'
    
    def test_parse_financial_value(self, extractor):
        """Test parsing financial values from text."""
        test_cases = [
            ("$90,151", 90151),
            ("$394,328 million", 394328),
            ("23,464", 23464),
            ("$96.9 billion", 96900),
            ("1,234.5 million", 1234.5)
        ]
        
        for text, expected in test_cases:
            result = extractor._parse_financial_value(text)
            assert result == expected
    
    def test_calculate_confidence_score(self, extractor):
        """Test confidence score calculation."""
        # Test high confidence case
        high_confidence_text = "Total net sales: $90,151 million"
        confidence = extractor._calculate_confidence_score(high_confidence_text, "revenue")
        assert confidence > 0.8
        
        # Test low confidence case
        low_confidence_text = "Some random text without clear financial data"
        confidence = extractor._calculate_confidence_score(low_confidence_text, "revenue")
        assert confidence < 0.5


class TestDocumentProcessingIntegration:
    """Integration tests for document processing with real files."""
    
    @pytest.mark.integration
    def test_full_document_processing_pipeline(self, test_settings):
        """Test complete document processing pipeline."""
        processor = FinancialDocumentProcessor()
        extractor = SECDataExtractor()
        
        # Create a comprehensive test document
        test_content = """
        APPLE INC.
        CONSOLIDATED STATEMENTS OF OPERATIONS
        (In millions, except per share amounts)
        
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
        
        CONSOLIDATED BALANCE SHEETS
        September 24, 2022
        ASSETS
        Current assets:
        Cash and cash equivalents    $ 48,844
        Total current assets         106,392
        
        CONSOLIDATED STATEMENTS OF CASH FLOWS
        Year Ended September 24, 2022
        Cash generated by operating activities    $ 122,151
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file_path = f.name
        
        try:
            # Test document processing
            metadata = {
                'ticker': 'AAPL',
                'filing_type': '10-K',
                'filing_date': '2022-09-24',
                'source': 'SEC Edgar'
            }
            
            # Process for chunks
            chunks = processor.process_sec_filing(test_file_path, metadata)
            assert len(chunks) > 0
            
            # Process for metrics
            metrics = extractor.process_document(test_file_path, "AAPL")
            assert len(metrics) > 0
            
            # Verify we have both chunks and metrics
            assert len(chunks) > 0
            assert len(metrics) > 0
            
            # Verify chunk content contains financial data
            chunk_content = " ".join([chunk.page_content for chunk in chunks])
            assert "revenue" in chunk_content.lower() or "sales" in chunk_content.lower()
            assert "income" in chunk_content.lower()
            
            # Verify metrics contain expected financial data
            metric_names = [m['metric_name'] for m in metrics]
            assert any('revenue' in name.lower() or 'sales' in name.lower() for name in metric_names)
            assert any('income' in name.lower() for name in metric_names)
        
        finally:
            Path(test_file_path).unlink() 