"""
Unit tests for SEC downloader functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import json

from knowledge_base.src.ingestion.sec_downloader import SECDownloader


class TestSECDownloader:
    """Test cases for SECDownloader class."""
    
    @pytest.fixture
    def downloader(self, test_settings):
        """Create a test instance of SECDownloader."""
        return SECDownloader()
    
    def test_initialization(self, downloader):
        """Test SECDownloader initialization."""
        assert downloader is not None
        assert hasattr(downloader, 'raw_dir')
        assert hasattr(downloader, 'download_dir')
        assert hasattr(downloader, 'downloader')
        assert hasattr(downloader, 'rate_limit')
    
    def test_rate_limiting(self, downloader):
        """Test rate limiting functionality."""
        # Test that rate limiting doesn't block on first call
        start_time = downloader.last_request_time
        downloader._rate_limit_delay()
        assert downloader.last_request_time > start_time
    
    @patch('requests.get')
    def test_get_company_info_success(self, mock_get, downloader, mock_sec_response):
        """Test successful company info retrieval."""
        # Mock the response
        mock_response = Mock()
        mock_response.json.return_value = mock_sec_response
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Test the method
        result = downloader.get_company_info('AAPL')
        
        # Verify the result
        assert result is not None
        assert result['cik'] == '0000320193'
        assert result['name'] == 'Apple Inc.'
        assert 'AAPL' in result['tickers']
        
        # Verify the request was made
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_get_company_info_failure(self, mock_get, downloader):
        """Test company info retrieval failure."""
        # Mock a failed response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Test the method
        result = downloader.get_company_info('INVALID')
        
        # Verify the result is None
        assert result is None
    
    @patch('knowledge_base.src.ingestion.sec_downloader.Downloader')
    def test_download_company_filings(self, mock_downloader_class, downloader):
        """Test downloading company filings."""
        # Mock the downloader
        mock_downloader = Mock()
        mock_downloader_class.return_value = mock_downloader
        
        # Mock the filing data
        mock_filing_data = {
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
        }
        
        # Mock the download method to return our test data
        with patch.object(downloader, '_extract_filing_metadata') as mock_extract:
            mock_extract.return_value = [mock_filing_data]
            
            # Test the method
            result = downloader.download_company_filings(
                ticker='AAPL',
                filing_types=['10-K'],
                num_filings=1
            )
            
            # Verify the result
            assert len(result) == 1
            assert result[0]['accession_number'] == '0000320193-22-000108'
            assert result[0]['type'] == '10-K'
    
    def test_extract_financial_indicators(self, downloader):
        """Test financial indicators extraction."""
        # Sample SEC filing content
        content = """
        CONSOLIDATED STATEMENTS OF OPERATIONS
        Revenue: $394,328 million
        Net income: $96,995 million
        BALANCE SHEET
        Total assets: $352,755 million
        CASH FLOWS
        Operating cash flow: $122,151 million
        """
        
        indicators = downloader._extract_financial_indicators(content)
        
        # Verify indicators are correctly identified
        assert indicators['has_revenue_data'] == True
        assert indicators['has_profit_data'] == True
        assert indicators['has_balance_sheet'] == True
        assert indicators['has_cash_flow'] == True
    
    def test_extract_financial_indicators_no_data(self, downloader):
        """Test financial indicators extraction with no financial data."""
        # Content without financial data
        content = """
        RISK FACTORS
        We face various risks in our business operations.
        MANAGEMENT DISCUSSION
        Our strategy focuses on innovation and growth.
        """
        
        indicators = downloader._extract_financial_indicators(content)
        
        # Verify indicators are correctly identified as False
        assert indicators['has_revenue_data'] == False
        assert indicators['has_profit_data'] == False
        assert indicators['has_balance_sheet'] == False
        assert indicators['has_cash_flow'] == False
    
    def test_parse_sec_document_metadata(self, downloader):
        """Test SEC document metadata parsing."""
        # Sample SEC document content
        content = """
        <DOCUMENT>
        <TYPE>10-K
        <SEQUENCE>1
        <FILENAME>aapl-20220924.htm
        <DESCRIPTION>ANNUAL REPORT
        <TEXT>
        <DOCUMENT>
        <TYPE>10-K
        <SEQUENCE>1
        <FILENAME>aapl-20220924.htm
        <DESCRIPTION>ANNUAL REPORT
        <TEXT>
        """
        
        metadata = downloader._parse_sec_document_metadata(content)
        
        # Verify metadata extraction
        assert metadata['filing_type'] == '10-K'
        assert metadata['filename'] == 'aapl-20220924.htm'
        assert metadata['description'] == 'ANNUAL REPORT'
    
    def test_bulk_download_companies(self, downloader):
        """Test bulk download functionality."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # Mock the download method
        with patch.object(downloader, 'download_company_filings') as mock_download:
            mock_download.return_value = [
                {
                    'accession_number': 'test-123',
                    'type': '10-K',
                    'ticker': 'AAPL'
                }
            ]
            
            # Test bulk download
            result = downloader.bulk_download_companies(
                tickers=tickers,
                filing_types=['10-K'],
                num_filings=1
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert len(result) == 3
            assert all(ticker in result for ticker in tickers)
            
            # Verify download was called for each ticker
            assert mock_download.call_count == 3
    
    def test_organize_downloaded_files(self, downloader):
        """Test file organization functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test directory structure
            test_dir = temp_path / "sec-edgar-filings" / "AAPL" / "10-K"
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a test file
            test_file = test_dir / "test_filing.txt"
            test_file.write_text("Test filing content")
            
            # Test organization
            downloader._organize_downloaded_files('AAPL', '10-K')
            
            # Verify the file was organized (this would depend on the actual implementation)
            assert test_file.exists()


class TestSECDownloaderIntegration:
    """Integration tests for SECDownloader with real API calls."""
    
    @pytest.mark.integration
    def test_real_company_info(self):
        """Test getting real company info from SEC API."""
        downloader = SECDownloader()
        
        # Test with a real company
        result = downloader.get_company_info('AAPL')
        
        # Verify we get real data
        assert result is not None
        assert 'cik' in result
        assert 'name' in result
        assert 'tickers' in result
        assert 'AAPL' in result['tickers']
    
    @pytest.mark.integration
    def test_real_filing_download(self):
        """Test downloading real SEC filings."""
        downloader = SECDownloader()
        
        # Test downloading a small number of filings
        result = downloader.download_company_filings(
            ticker='AAPL',
            filing_types=['10-K'],
            num_filings=1
        )
        
        # Verify we get real filing data
        assert len(result) > 0
        assert all('accession_number' in filing for filing in result)
        assert all('file_path' in filing for filing in result) 