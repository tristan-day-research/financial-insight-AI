"""
SEC EDGAR API client for downloading financial filings.
Handles rate limiting, authentication, and file processing.
"""

import requests
import time
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import re
from urllib.parse import urljoin, urlparse
import zipfile
import io
from sec_edgar_downloader import Downloader
# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config.settings import get_settings

logger = logging.getLogger(__name__)


class SECDownloader:
    """Downloads and processes SEC filings with rate limiting and metadata extraction."""

    def __init__(self):
        self.settings = get_settings()
        
        # Set up the raw data directory
        self.raw_dir = Path(self.settings.data.raw_data_path)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary download directory for sec-edgar-filings
        self.download_dir = self.raw_dir / "sec-edgar-filings"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SEC downloader with our settings
        self.downloader = Downloader(
            self.settings.api.sec_user_agent,
            self.settings.api.sec_email,
            download_folder=str(self.raw_dir)  # Use raw_dir as base, library will create sec-edgar-filings
        )
        
        # Set rate limiting
        self.rate_limit = self.settings.api.max_requests_per_minute
        self.last_request_time = 0

            # Add headers for SEC API requests
        self.headers = {
            "User-Agent": f"{self.settings.api.sec_user_agent} {self.settings.api.sec_email}",
            "Accept-Encoding": "gzip, deflate"
        }
        
        # Different headers for data.sec.gov
        self.data_headers = {
            "User-Agent": f"{self.settings.api.sec_user_agent} {self.settings.api.sec_email}",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
    
    def _migrate_old_structure(self, old_structure: Path):
        """Migrate files from old sec-edgar-filings structure to new flat structure."""
        try:
            # Iterate through company directories
            for company_dir in old_structure.iterdir():
                if company_dir.is_dir():
                    # For each filing type directory
                    for filing_type_dir in company_dir.iterdir():
                        if filing_type_dir.is_dir():
                            # Create new directory structure
                            new_dir = self.download_dir / company_dir.name / filing_type_dir.name
                            new_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Move all contents
                            for item in filing_type_dir.glob("*"):
                                if item.is_dir():
                                    target_dir = new_dir / item.name
                                    if not target_dir.exists():
                                        item.rename(target_dir)
                            
                            # Try to remove old directories if empty
                            try:
                                filing_type_dir.rmdir()
                            except OSError:
                                pass
                    
                    try:
                        company_dir.rmdir()
                    except OSError:
                        pass
            
            # Try to remove the root sec-edgar-filings directory
            try:
                old_structure.rmdir()
            except OSError:
                pass
                
        except Exception as e:
            logger.warning(f"Error during directory migration: {str(e)}")
        
    def _rate_limit_delay(self):
        """Implement rate limiting for SEC API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _organize_downloaded_files(self, ticker: str, filing_type: str) -> None:
        """Move downloaded files from sec-edgar-filings to our organized structure."""
        # Source directory (where the SEC library downloads files)
        src_dir = self.raw_dir / "sec-edgar-filings" / ticker / filing_type
        if not src_dir.exists():
            logger.warning(f"Source directory not found: {src_dir}")
            return

        # Target directory in our organized structure
        target_dir = self.raw_dir / ticker / filing_type
        target_dir.mkdir(parents=True, exist_ok=True)

        # Move each filing to its proper location
        moved_count = 0
        for filing_dir in src_dir.glob("*"):
            if filing_dir.is_dir():
                new_location = target_dir / filing_dir.name
                if not new_location.exists():
                    try:
                        # First, find and rename XBRL and HTML files to make them easier to find
                        for root, _, files in os.walk(filing_dir):
                            for file in files:
                                file_path = Path(root) / file
                                if file.endswith('.xml') or file.endswith('.xbrl'):
                                    # Move XBRL file to filing directory with clear name
                                    new_name = filing_dir.name + '_xbrl.xml'
                                    new_path = filing_dir / new_name
                                    file_path.rename(new_path)
                                    logger.info(f"Renamed XBRL file to {new_name}")
                                elif file.endswith('.htm') or file.endswith('.html'):
                                    # Move HTML file to filing directory with clear name
                                    new_name = filing_dir.name + '.html'
                                    new_path = filing_dir / new_name
                                    file_path.rename(new_path)
                                    logger.info(f"Renamed HTML file to {new_name}")

                        # Move the entire directory
                        filing_dir.rename(new_location)
                        moved_count += 1
                        logger.info(f"Moved filing {filing_dir.name} to {new_location}")
                    except Exception as e:
                        logger.error(f"Error moving filing {filing_dir.name}: {str(e)}")

        if moved_count > 0:
            logger.info(f"Moved {moved_count} filings to organized structure")
            
            # Clean up empty directories
            try:
                src_dir.rmdir()  # Remove filing type directory
                src_dir.parent.rmdir()  # Remove ticker directory
                self.download_dir.rmdir()  # Remove sec-edgar-filings
            except OSError:
                pass  # Ignore if directories aren't empty

    def download_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = None,
        num_filings: int = 5,
        after_date: Optional[str] = None
    ) -> List[Dict]:
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K']
        
        downloaded_filings = []

        print(ticker, filing_types, num_filings)
        
        # Get company info
        company_info = self.get_company_info(ticker)
        if not company_info:
            logger.error(f"Could not find company info for {ticker}")
            return []
            
        cik = company_info['cik']
        
        for filing_type in filing_types:
            try:
                logger.info(f"Downloading {filing_type} filings for {ticker}")
                self._rate_limit_delay()
                
                # Get filing metadata from SEC API
                # Remove any leading zeros and then pad to 10 digits
                cik_stripped = str(int(cik))  # Remove leading zeros
                cik_padded = cik_stripped.zfill(10)  # Pad to 10 digits
                url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
                response = requests.get(url, headers=self.data_headers)
                if response.status_code != 200:
                    logger.error(f"Error getting filings: {response.status_code}")
                    continue
                    
                data = response.json()
                
                # Log the raw response for debugging
                logger.debug("SEC API Response: %s", json.dumps(data, indent=2))
                
                # Check if we have the expected structure
                if 'filings' not in data or 'recent' not in data['filings']:
                    logger.error("Unexpected API response structure")
                    continue
                    
                filings = data['filings']['recent']
                
                # Get company information including fiscal year end
                company_fiscal_year_end = None
                if 'fiscalYearEnd' in data:
                    fiscal_year_end_code = data['fiscalYearEnd']
                    # Convert MMYY format to YYYY-MM-DD
                    if len(fiscal_year_end_code) == 4:
                        month = fiscal_year_end_code[:2]
                        day = fiscal_year_end_code[2:]
                        # Use current year as base, but we'll adjust based on report date
                        company_fiscal_year_end = f"20XX-{month}-{day}"
                        logger.info(f"Company fiscal year end: {company_fiscal_year_end} (from code: {fiscal_year_end_code})")
                
                # Validate we have the required data
                if not filings or not isinstance(filings, dict):
                    logger.error("No filings data found or invalid format")
                    continue
                    
                # Log the filings structure
                logger.debug("Filings structure: %s", json.dumps(filings, indent=2))
                
                # Check what fields are actually available
                logger.info("Available fields in filings: %s", list(filings.keys()))
                if 'fiscalYearEnd' in filings:
                    logger.info("fiscalYearEnd data: %s", filings['fiscalYearEnd'])
                else:
                    logger.warning("fiscalYearEnd field not found in filings data")
                    
                # Check the first few entries of each field to understand the structure
                for field in ['form', 'reportDate', 'accessionNumber', 'fiscalYearEnd']:
                    if field in filings:
                        logger.info("%s (first 3 entries): %s", field, filings[field][:3])
                    else:
                        logger.warning("Field '%s' not found in filings data", field)
                
                # Filter for filing type
                if 'form' not in filings:
                    logger.error("'form' field not found in filings data")
                    continue
                    
                form_indices = [i for i, form in enumerate(filings['form']) if form == filing_type]
                if not form_indices:
                    logger.warning(f"No {filing_type} filings found for {ticker}")
                    continue
                    
                # Take only requested number of filings
                form_indices = form_indices[:num_filings]
                
                # Create target directory
                company_dir = self.raw_dir / ticker / filing_type
                company_dir.mkdir(parents=True, exist_ok=True)
                
                # Download each filing
                for idx in form_indices:
                    # Check for required fields
                    if 'accessionNumber' not in filings:
                        logger.error("'accessionNumber' field not found in filings data")
                        continue
                    if 'reportDate' not in filings:
                        logger.error("'reportDate' field not found in filings data")
                        continue
                        
                    accession = filings['accessionNumber'][idx].replace('-', '')
                    
                    # Get the filing index page
                    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
                    self._rate_limit_delay()
                    response = requests.get(index_url, headers=self.headers)
                    if response.status_code != 200:
                        logger.error(f"Error getting filing index: {response.status_code}")
                        continue
                        
                    index_data = response.json()
                    
                    # Find the main filing document, XBRL, and HTML files
                    main_doc = None
                    xbrl_doc = None
                    html_doc = None
                    
                    # Get the list of files from the index
                    files = index_data.get('directory', {}).get('item', [])
                    
                    # Sometimes 'item' is a dict instead of a list for single files
                    if isinstance(files, dict):
                        files = [files]
                    
                    for file in files:
                        name = file.get('name', '').lower()
                        if not name:
                            continue
                            
                        if name.endswith('.htm'):
                            html_doc = file['name']
                        elif name.endswith('.xml') and 'xbrl' in name:
                            xbrl_doc = file['name']
                        # Main doc is usually the first .txt file
                        elif name.endswith('.txt'):
                            main_doc = file['name']
                    
                    # Download each file type
                    filing_dir = company_dir / accession
                    filing_dir.mkdir(parents=True, exist_ok=True)
                    
                    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
                    
                    if main_doc:
                        self._download_file(base_url + main_doc, filing_dir / 'full-submission.txt')
                    if xbrl_doc:
                        self._download_file(base_url + xbrl_doc, filing_dir / f"{accession}_xbrl.xml")
                    if html_doc:
                        self._download_file(base_url + html_doc, filing_dir / f"{accession}.html")
                    
                    # Create simple metadata (old working version style)
                    # Get actual fiscal year end from SEC API data
                    actual_fiscal_year_end = None
                    if company_fiscal_year_end:
                        # Extract the month and day from the company fiscal year end
                        # and combine with the report date year
                        try:
                            from datetime import datetime
                            report_date_obj = datetime.strptime(filings['reportDate'][idx], '%Y-%m-%d')
                            
                            # Parse the company fiscal year end (format: "20XX-MM-DD")
                            if company_fiscal_year_end.startswith("20XX-"):
                                month_day = company_fiscal_year_end[5:]  # Get MM-DD part
                                actual_fiscal_year_end = f"{report_date_obj.year}-{month_day}"
                            else:
                                actual_fiscal_year_end = company_fiscal_year_end
                                
                            logger.info(f"Using actual fiscal year end: {actual_fiscal_year_end}")
                        except Exception as e:
                            logger.warning(f"Error parsing fiscal year end: {e}")
                            actual_fiscal_year_end = None
                    
                    # Fallback to inference if actual data not available
                    if not actual_fiscal_year_end:
                        actual_fiscal_year_end = self._infer_fiscal_year_end(
                            filings['reportDate'][idx], 
                            company_info['title']
                        )
                    
                    metadata = {
                        'ticker': ticker,
                        'type': filing_type,
                        'file_path': str(filing_dir / 'full-submission.txt'),
                        'date': filings['reportDate'][idx],
                        'downloaded_at': datetime.now().isoformat(),
                        'file_size': os.path.getsize(filing_dir / 'full-submission.txt'),
                        'accession_number': filings['accessionNumber'][idx],
                        'period_of_report': filings['reportDate'][idx],
                        'company_name': company_info['title'],
                        'cik': str(cik),
                        'fiscal_year_end': actual_fiscal_year_end,
                        'business_address': '',
                        'has_revenue_data': True,
                        'has_profit_data': True,
                        'has_balance_sheet': True,
                        'has_cash_flow': True,
                        'financial_keywords_count': 0,
                        'metadata_path': str(filing_dir / 'metadata.json')
                    }
                    
                    # Log the filings data structure for debugging
                    logger.debug("Filings data structure: %s", json.dumps(filings, indent=2))
                    
                    # Remove None values from metadata
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                    
                    # Save metadata
                    with open(filing_dir / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    downloaded_filings.append(metadata)
                    
                logger.info(f"Downloaded {len(form_indices)} {filing_type} filings for {ticker}")
                
            except Exception as e:
                print("Exception in download_company_filings")
                logger.error(f"Error downloading {filing_type} for {ticker}: {str(e)}")
                continue
        
        return downloaded_filings
        
    def _download_file(self, url: str, target_path: Path) -> None:
        """Download a file from SEC EDGAR."""
        self._rate_limit_delay()
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {target_path.name}")
        else:
            logger.error(f"Error downloading {url}: {response.status_code}")
    
    def _get_relative_path(self, path: Path) -> str:
        """Convert absolute path to relative path from data directory."""
        try:
            return str(path.relative_to(self.download_dir))
        except ValueError:
            return str(path)

    def _save_metadata_json(self, metadata: Dict, filing_path: Path) -> Path:
        """Save filing metadata to a JSON file alongside the raw document."""
        # Get the document's directory (where the raw file is stored)
        doc_dir = Path(filing_path).parent
        
        # Create a unique metadata filename based on accession number
        metadata_filename = f"metadata.json"
        metadata_path = doc_dir / metadata_filename
        
        # Get relative paths
        relative_doc_path = self._get_relative_path(filing_path)
        relative_metadata_path = self._get_relative_path(metadata_path)
        
        # Enhanced metadata for federated knowledge base
        enhanced_metadata = {
            # Document identification
            "doc_id": metadata["accession_number"],  # Unique identifier
            "ticker": metadata["ticker"],
            "filing_type": metadata["type"],
            "filing_date": metadata.get("period_of_report"),
            "accession_number": metadata["accession_number"],
            
            # File information
            "raw_file": {
                "path": relative_doc_path,
                "filename": filing_path.name,
                "size_bytes": metadata.get("file_size"),
                "last_modified": datetime.now().isoformat()
            },
            
            # Company information
            "company": {
                "name": metadata.get("company_name"),
                "cik": metadata.get("cik"),
                "fiscal_year_end": metadata.get("fiscal_year_end"),
                "business_address": metadata.get("business_address")
            },
            
            # Document metadata
            "document": {
                "title": f"{metadata['ticker']} {metadata['type']} - {metadata.get('period_of_report')}",
                "source": "SEC EDGAR",
                "language": "en",
                "doc_type": "financial_filing",
                "filing_period": metadata.get("period_of_report"),
                "document_date": metadata.get("date")
            },
            
            # Processing metadata
            "processing": {
                "downloaded_at": metadata.get("downloaded_at"),
                "download_agent": "Financial Insight AI",
                "download_source": "SEC EDGAR API",
                "version": "1.0"
            },
            
            # Knowledge base integration
            "knowledge_base": {
                "vector_store_id": None,  # To be filled by vector store
                "sql_record_id": None,    # To be filled by SQL store
                "graph_node_id": None,    # To be filled by knowledge graph
                "relationships": [],      # To be filled with related documents
                "tags": [
                    metadata["type"],
                    metadata["ticker"],
                    f"FY{metadata.get('fiscal_year_end', '')}"
                ]
            }
        }
        
        # Save metadata to JSON file
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
            
        return metadata_path

    def _extract_filing_metadata(
        self, 
        ticker: str, 
        filing_type: str, 
        count: int,
        download_dir: Path
    ) -> List[Dict]:
        """Extract metadata from files in the specified download directory and save to JSON."""
        print("_extract_filing_metadata called")
        print("download_dir", download_dir)
        metadata_list = []
        
        for filing_path in download_dir.rglob("*.txt"):  # Or the correct file extension
            print("filing_path", filing_path)
            
            # Extract basic metadata
            metadata = {
                "ticker": ticker,
                "type": filing_type,
                "file_path": str(filing_path),
                "date": filing_path.stem.split("_")[-1],  # Adjust based on filename format
                "downloaded_at": datetime.now().isoformat(),
                "file_size": filing_path.stat().st_size,
                "accession_number": filing_path.parent.name  # The folder name is usually the accession number
            }
            
            # Parse document content for additional metadata
            try:
                with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    doc_metadata = self._parse_sec_document_metadata(content)
                    metadata.update(doc_metadata)
            except Exception as e:
                logger.warning(f"Error parsing document content for metadata: {str(e)}")
            
            # Save metadata to JSON file
            metadata_path = self._save_metadata_json(metadata, filing_path)
            metadata["metadata_path"] = str(metadata_path)
            
            metadata_list.append(metadata)
            
        return metadata_list
    
    def _parse_sec_document_metadata(self, content: str) -> Dict:
        """Parse SEC document content to extract metadata."""
        metadata = {}
        
        try:
            # Look for common SEC document headers
            lines = content.split('\n')[:100]  # Check first 100 lines
            
            for line in lines:
                line = line.strip()
                
                # Extract company name
                if "COMPANY CONFORMED NAME:" in line:
                    metadata["company_name"] = line.split(":")[-1].strip()
                
                # Extract CIK
                elif "CENTRAL INDEX KEY:" in line:
                    metadata["cik"] = line.split(":")[-1].strip()
                
                # Extract fiscal year end
                elif "FISCAL YEAR END:" in line:
                    metadata["fiscal_year_end"] = line.split(":")[-1].strip()
                
                # Extract period of report
                elif "CONFORMED PERIOD OF REPORT:" in line:
                    metadata["period_of_report"] = line.split(":")[-1].strip()
                
                # Extract business address
                elif "BUSINESS ADDRESS:" in line:
                    # This usually spans multiple lines, simplified extraction
                    metadata["business_address"] = line.split(":")[-1].strip()
            
            # Try to extract key financial terms for enhanced searchability
            financial_indicators = self._extract_financial_indicators(content)
            metadata.update(financial_indicators)
            
        except Exception as e:
            logger.warning(f"Error parsing SEC document metadata: {str(e)}")
        
        return metadata
    
    def _extract_fiscal_year_end_from_date(self, report_date: str) -> str:
        """
        Extract fiscal year end from report date.
        Most companies have fiscal year end in September (09-30), but some use December (12-31).
        """
        try:
            from datetime import datetime
            date_obj = datetime.strptime(report_date, '%Y-%m-%d')
            
            # Check if the report date is in Q4 (Oct-Dec), then use that year
            # Otherwise, use the previous year
            if date_obj.month >= 10:  # October or later
                fiscal_year = date_obj.year
            else:
                fiscal_year = date_obj.year - 1
                
            # Most companies use 09-30, but some use 12-31
            # For now, assume 09-30 as it's more common
            fiscal_year_end = f"{fiscal_year}-09-30"
            return fiscal_year_end
        except Exception as e:
            logger.warning("Could not extract fiscal year end from report date %s: %s", report_date, str(e))
            return None

    def _infer_fiscal_year_end(self, report_date: str, company_name: str = None) -> str:
        """
        Infer fiscal year end from report date and company information.
        Uses common patterns and company-specific rules.
        """
        if not report_date:
            return None
            
        try:
            from datetime import datetime
            date_obj = datetime.strptime(report_date, '%Y-%m-%d')
            
            # Determine fiscal year
            if date_obj.month >= 10:  # October or later
                fiscal_year = date_obj.year
            else:
                fiscal_year = date_obj.year - 1
            
            # Company-specific fiscal year end patterns
            company_fiscal_ends = {
                'APPLE': '09-30',      # Apple uses September 30
                'MICROSOFT': '06-30',  # Microsoft uses June 30
                'GOOGLE': '12-31',     # Google uses December 31
                'ALPHABET': '12-31',   # Alphabet (Google parent) uses December 31
                'MICROSTRATEGY': '12-31', # MicroStrategy uses December 31
                'AMAZON': '12-31',     # Amazon uses December 31
                'META': '12-31',       # Meta uses December 31
                'NETFLIX': '12-31',    # Netflix uses December 31
                'TESLA': '12-31',      # Tesla uses December 31
                'NVIDIA': '01-31',     # NVIDIA uses January 31
                'ADOBE': '11-30',      # Adobe uses November 30
                'SALESFORCE': '01-31', # Salesforce uses January 31
                'ORACLE': '05-31',     # Oracle uses May 31
                'INTEL': '12-31',      # Intel uses December 31
                'CISCO': '07-31',      # Cisco uses July 31
            }
            
            # Try to match company name
            if company_name:
                company_upper = company_name.upper()
                for company_key, fiscal_end in company_fiscal_ends.items():
                    if company_key in company_upper:
                        return f"{fiscal_year}-{fiscal_end}"
            
            # Default patterns based on industry/common practices
            # Most tech companies use December 31
            # Most retail companies use January 31
            # Most manufacturing companies use September 30
            
            # For now, use December 31 as it's most common for large companies
            return f"{fiscal_year}-12-31"
            
        except Exception as e:
            logger.warning("Could not infer fiscal year end from report date %s: %s", report_date, str(e))
            return None

    def _extract_financial_indicators(self, content: str) -> Dict:
        """Extract financial indicators and metrics from document content."""
        indicators = {
            "has_revenue_data": False,
            "has_profit_data": False,
            "has_balance_sheet": False,
            "has_cash_flow": False,
            "financial_keywords_count": 0
        }
        
        content_lower = content.lower()
        financial_keywords = self.settings.data.financial_keywords
        
        # Count financial keywords
        keyword_count = sum(content_lower.count(keyword.lower()) for keyword in financial_keywords)
        indicators["financial_keywords_count"] = keyword_count
        
        # Check for specific financial statement types
        if any(term in content_lower for term in ["revenue", "net sales", "total revenue"]):
            indicators["has_revenue_data"] = True
            
        if any(term in content_lower for term in ["net income", "profit", "earnings"]):
            indicators["has_profit_data"] = True
            
        if any(term in content_lower for term in ["balance sheet", "assets", "liabilities"]):
            indicators["has_balance_sheet"] = True
            
        if any(term in content_lower for term in ["cash flow", "operating activities", "investing activities"]):
            indicators["has_cash_flow"] = True
        
        return indicators
    
    def bulk_download_companies(
        self,
        tickers: List[str],
        filing_types: List[str] = None,
        num_filings: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Download filings for multiple companies.
        
        Args:
            tickers: List of company ticker symbols
            filing_types: List of filing types to download
            num_filings: Number of each filing type per company
            
        Returns:
            Dictionary mapping ticker to list of filing metadata
        """
        if filing_types is None:
            filing_types = ['10-K', '10-Q']
        
        results = {}
        
        for ticker in tickers:
            logger.info(f"Starting bulk download for {ticker}")
            try:
                filings = self.download_company_filings(
                    ticker=ticker,
                    filing_types=filing_types,
                    num_filings=num_filings
                )
                results[ticker] = filings
                logger.info(f"Completed download for {ticker}: {len(filings)} filings")
                
                # Add delay between companies to be respectful to SEC servers
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to download filings for {ticker}: {str(e)}")
                results[ticker] = []
        
        return results
    
    def get_company_info(self, ticker: str) -> Optional[Dict]:
        """
        Get basic company information from SEC database.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Dictionary with company information or None if not found
        """
        try:
            # Get company info from SEC's company tickers API
            self._rate_limit_delay()
            
            # Use the correct API endpoint and headers
            headers = {
                "User-Agent": f"{self.settings.api.sec_user_agent} {self.settings.api.sec_email}",
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov"
            }
            
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=headers
            )
            if response.status_code != 200:
                logger.error(f"Error getting company tickers: {response.status_code}")
                return None
                
            companies = response.json()
            
            # Find the matching company (data structure is different in this endpoint)
            for company in companies.values():
                if company['ticker'] == ticker:
                    return {
                        "ticker": company['ticker'],
                        "title": company['title'],
                        "cik": str(company['cik_str']).zfill(10)  # Pad CIK to 10 digits
                    }
            
            logger.error(f"Company {ticker} not found in SEC database")
            return None
            
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {str(e)}")
            return None


def main():
    """Example usage of SEC downloader."""
    downloader = SECDownloader()
    
    # Download sample filings
    companies = ["AAPL", "MSFT", "GOOGL"]
    results = downloader.bulk_download_companies(
        tickers=companies,
        filing_types=["10-K", "10-Q"],
        num_filings=2
    )
    
    # Print summary
    for ticker, filings in results.items():
        print(f"\n{ticker}: Downloaded {len(filings)} filings")
        for filing in filings:
            print(f"  - {filing['filing_type']} from {filing['filing_date']}")


if __name__ == "__main__":
    main()

