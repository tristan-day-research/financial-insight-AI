"""
SEC Financial Data Extractor
Extracts structured financial metrics from SEC filings with maximum fidelity.
This module focuses purely on data extraction - database management is handled by sql_manager.py
"""

'''

Hybrid approach with XBRL priority
Why this approach:

Maximum Fidelity: XBRL provides exact, audited financial data
Fallback Robustness: PDF parsing when XBRL unavailable
Vector DB Synergy: Extract precise numbers for SQL, contextual narratives for vectors
Regulatory Compliance: XBRL is the official structured forma


This extraction code prioritizes fidelity and accuracy through:
Key Features:

Hierarchical Extraction: XBRL (98% confidence) → PDF Tables (85%) → Text Patterns (70-75%)
Maximum Fidelity Preservation:

Stores raw text alongside extracted values
Tracks extraction method and confidence
Preserves original context for audit trails


Robust Error Handling:

Multiple fallback methods
Validation checks for accounting consistency
Confidence scoring for each extraction


Production-Ready Features:

Database transactions
Duplicate handling
Standardized metric naming
Period normalization

'''

import re
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
import pdfplumber
import pandas as pd
from decimal import Decimal
import decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class FinancialMetric:
    """Represents a single financial metric with metadata"""
    company_id: str
    metric_name: str
    value: Decimal
    period: str
    unit: str
    source_document: str
    extraction_method: str
    confidence: float
    raw_text: Optional[str] = None

class SECDataExtractor:
    """Extract structured financial data from SEC filings"""
    
    def __init__(self):
        # Removed database initialization - handled by sql_manager
        
        # Standard GAAP metric mappings
        self.gaap_mappings = {
            'us-gaap:Revenues': 'revenue',
            'us-gaap:NetIncomeLoss': 'net_income',
            'us-gaap:Assets': 'total_assets',
            'us-gaap:Liabilities': 'total_liabilities',
            'us-gaap:StockholdersEquity': 'shareholders_equity',
            'us-gaap:CashAndCashEquivalentsAtCarryingValue': 'cash_equivalents',
            'us-gaap:ResearchAndDevelopmentExpense': 'rd_expense',
            'us-gaap:OperatingIncomeLoss': 'operating_income'
        }
        # Common financial statement patterns for PDF fallback
        self.financial_patterns = {
            'revenue': [
                r'(?:net\s+)?(?:revenues?|sales|net\s+sales)\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'total\s+(?:net\s+)?revenues?\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'(?:revenue|sales)\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?'
            ],
            'net_income': [
                r'net\s+(?:income|earnings?)\s*[:\s]*\$?\s*([\d,\-]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'(?:net\s+)?(?:income|earnings?)\s+(?:loss\s+)?attributable.*?\$?\s*([\d,\-]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'net\s+(?:income|earnings?)\s*[:\s]*\$?\s*([\d,\-]+(?:\.\d+)?)'
            ],
            'total_assets': [
                r'total\s+assets\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'assets\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?'
            ],
            'total_liabilities': [
                r'total\s+liabilities\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'liabilities\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?'
            ],
            'shareholders_equity': [
                r'(?:shareholders?|stockholders?)\s+equity\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'equity\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?'
            ]
        }

    # Database setup moved to sql_manager.py in the storage module

    def process_document(self, file_path: str, company_id: str) -> List[FinancialMetric]:
        """Main entry point - process a single SEC document"""
        file_path = Path(file_path)
        
        print(f"Processing {file_path.name} for {company_id}")
        
        # Try XBRL first (highest fidelity)
        xbrl_path = self._find_xbrl_file(file_path)
        if xbrl_path and xbrl_path.exists():
            metrics = self._extract_from_xbrl(xbrl_path, company_id, file_path.name)
            if metrics:
                print(f"Extracted {len(metrics)} metrics from XBRL")
                return metrics
        
        # Fallback to PDF extraction
        if file_path.suffix.lower() == '.pdf':
            metrics = self._extract_from_pdf(file_path, company_id)
            print(f"Extracted {len(metrics)} metrics from PDF")
            return metrics
        
        # Fallback to text extraction
        metrics = self._extract_from_text(file_path, company_id)
        print(f"Extracted {len(metrics)} metrics from text")
        return metrics

    def _find_xbrl_file(self, pdf_path: Path) -> Optional[Path]:
        """Look for corresponding XBRL file"""
        # XBRL files often accompany SEC filings
        xbrl_patterns = [
            pdf_path.with_suffix('.xml'),
            pdf_path.parent / f"{pdf_path.stem}_xbrl.xml",
            pdf_path.parent / f"{pdf_path.stem}.xbrl"
        ]
        
        for xbrl_path in xbrl_patterns:
            if xbrl_path.exists():
                return xbrl_path
        return None

    def _extract_from_xbrl(self, xbrl_path: Path, company_id: str, source_doc: str) -> List[FinancialMetric]:
        """Extract from XBRL file (highest fidelity method)"""
        metrics = []
        
        try:
            tree = ET.parse(xbrl_path)
            root = tree.getroot()
            
            # Define XBRL namespaces
            namespaces = {
                'xbrl': 'http://www.xbrl.org/2003/instance',
                'us-gaap': 'http://fasb.org/us-gaap/2023'
            }
            
            # Extract period information
            period = self._extract_period_from_xbrl(root, namespaces)
            
            for gaap_tag, metric_name in self.gaap_mappings.items():
                elements = root.findall(f".//{gaap_tag}", namespaces)
                
                for element in elements:
                    try:
                        value = Decimal(element.text.replace(',', ''))
                        unit = element.get('unitRef', 'USD')
                        
                        metric = FinancialMetric(
                            company_id=company_id,
                            metric_name=metric_name,
                            value=value,
                            period=period,
                            unit=unit,
                            source_document=source_doc,
                            extraction_method='xbrl',
                            confidence=0.98,  # XBRL is very reliable
                            raw_text=element.text
                        )
                        metrics.append(metric)
                        
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing XBRL value for {metric_name}: {e}")
                        continue
                        
        except ET.ParseError as e:
            print(f"Error parsing XBRL file {xbrl_path}: {e}")
            
        return metrics

    def _extract_period_from_xbrl(self, root, namespaces) -> str:
        """Extract reporting period from XBRL"""
        # Look for period information
        contexts = root.findall('.//xbrl:context', namespaces)
        for context in contexts:
            period_elem = context.find('.//xbrl:period', namespaces)
            if period_elem is not None:
                instant = period_elem.find('.//xbrl:instant', namespaces)
                if instant is not None:
                    date_str = instant.text
                    return self._standardize_period(date_str)
        
        return "unknown_period"

    def _extract_from_pdf(self, pdf_path: Path, company_id: str) -> List[FinancialMetric]:
        """Extract from PDF using table detection and pattern matching"""
        metrics = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # First, try to find financial statement tables
                tables_metrics = self._extract_from_tables(pdf, company_id, pdf_path.name)
                metrics.extend(tables_metrics)
                
                # Then, extract any additional metrics from text
                if len(metrics) < 5:  # If tables didn't yield much, try text patterns
                    text_metrics = self._extract_from_pdf_text(pdf, company_id, pdf_path.name)
                    metrics.extend(text_metrics)
                    
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            
        return metrics

    def _extract_from_tables(self, pdf, company_id: str, source_doc: str) -> List[FinancialMetric]:
        """Extract financial data from PDF tables"""
        metrics = []
        
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            
            for table in tables:
                if self._is_financial_table(table):
                    period = self._extract_period_from_table(table)
                    table_metrics = self._parse_financial_table(table, company_id, period, source_doc)
                    metrics.extend(table_metrics)
                    
        return metrics

    def _is_financial_table(self, table: List[List[str]]) -> bool:
        """Identify if a table contains financial data"""
        if not table or len(table) < 3:
            return False
            
        # Look for financial keywords in headers
        header_text = ' '.join([cell or '' for cell in table[0]]).lower()
        financial_keywords = ['revenue', 'income', 'assets', 'liabilities', 'cash', 'million', 'thousand']
        
        return any(keyword in header_text for keyword in financial_keywords)

    def _parse_financial_table(self, table: List[List[str]], company_id: str, 
                             period: str, source_doc: str) -> List[FinancialMetric]:
        """Parse financial metrics from a table"""
        metrics = []
        
        for row in table[1:]:  # Skip header
            if not row or len(row) < 2:
                continue
                
            metric_name = (row[0] or '').strip().lower()
            
            # Try to find a standardized metric name
            standardized_name = self._standardize_metric_name(metric_name)
            if not standardized_name:
                continue
                
            # Extract numeric values from subsequent columns
            for col_idx in range(1, len(row)):
                cell_value = row[col_idx]
                if cell_value:
                    numeric_value = self._extract_numeric_value(cell_value)
                    if numeric_value is not None:
                        metric = FinancialMetric(
                            company_id=company_id,
                            metric_name=standardized_name,
                            value=numeric_value,
                            period=period,
                            unit='USD',
                            source_document=source_doc,
                            extraction_method='pdf_table',
                            confidence=0.85,
                            raw_text=cell_value
                        )
                        metrics.append(metric)
                        break  # Take first valid numeric value
                        
        return metrics

    def _extract_from_pdf_text(self, pdf, company_id: str, source_doc: str) -> List[FinancialMetric]:
        """Extract from PDF text using regex patterns"""
        metrics = []
        
        # Combine text from all pages
        full_text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
        
        # Extract period from document
        period = self._extract_period_from_text(full_text)
        
        # Apply regex patterns
        for metric_name, patterns in self.financial_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    try:
                        raw_value = match.group(1)
                        numeric_value = self._extract_numeric_value(raw_value)
                        
                        if numeric_value is not None:
                            metric = FinancialMetric(
                                company_id=company_id,
                                metric_name=metric_name,
                                value=numeric_value,
                                period=period,
                                unit='USD',
                                source_document=source_doc,
                                extraction_method='pdf_regex',
                                confidence=0.75,
                                raw_text=match.group(0)
                            )
                            metrics.append(metric)
                            break  # Take first match for each metric
                            
                    except (ValueError, IndexError):
                        continue
                        
        return metrics

    def _extract_from_text(self, file_path: Path, company_id: str) -> List[FinancialMetric]:
        """Extract from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            print(f"Original text length: {len(text)} characters")
            
            # Check if this is an XBRL file
            if '<ix:nonFraction' in text or '<ix:nonNumeric' in text:
                print("Detected XBRL content, extracting from XBRL...")
                return self._extract_from_xbrl_text(text, company_id, file_path.name)
                
            # Clean HTML entities and tags if this appears to be HTML
            if file_path.suffix.lower() in ['.html', '.htm']:
                text = self._clean_html_text(text)
                print(f"Cleaned text length: {len(text)} characters")
                
            return self._extract_from_text_content(text, company_id, file_path.name)
            
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return []

    def _clean_html_text(self, text: str) -> str:
        """Clean HTML text for better extraction"""
        import html
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common HTML artifacts and non-breaking spaces
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        text = re.sub(r'\xa0', ' ', text)  # Remove non-breaking spaces
        
        # Remove page numbers and navigation elements
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'Table of Contents', '', text)
        
        # Remove XBRL tags that might interfere
        text = re.sub(r'<ix:[^>]+>', '', text)
        text = re.sub(r'</ix:[^>]+>', '', text)
        
        return text.strip()

    def _has_financial_content(self, text: str) -> bool:
        """Check if text contains financial content"""
        financial_keywords = [
            'revenue', 'income', 'assets', 'liabilities', 'equity', 'cash',
            'million', 'billion', 'thousand', 'dollars', 'financial',
            'balance sheet', 'income statement', 'cash flow'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in financial_keywords)

    def _extract_from_text_content(self, text: str, company_id: str, source_doc: str) -> List[FinancialMetric]:
        """Extract metrics from text content using patterns"""
        metrics = []
        
        # Check if text contains financial content
        if not self._has_financial_content(text):
            print(f"No financial content detected in {source_doc}")
            return metrics
            
        period = self._extract_period_from_text(text)
        
        for metric_name, patterns in self.financial_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    try:
                        raw_value = match.group(1)
                        
                        # Additional validation - ensure it's actually a number
                        if not re.match(r'^[\d,\-\.]+$', raw_value):
                            continue
                            
                        numeric_value = self._extract_numeric_value(raw_value)
                        
                        if numeric_value is not None:
                            metric = FinancialMetric(
                                company_id=company_id,
                                metric_name=metric_name,
                                value=numeric_value,
                                period=period,
                                unit='USD',
                                source_document=source_doc,
                                extraction_method='text_regex',
                                confidence=0.70,
                                raw_text=match.group(0)
                            )
                            metrics.append(metric)
                            break  # Take first match for each metric
                            
                    except (ValueError, IndexError):
                        continue
                        
        return metrics

    def _extract_numeric_value(self, text: str) -> Optional[Decimal]:
        """Extract and normalize numeric values"""
        if not text:
            return None
            
        # Remove HTML entities and common formatting
        clean_text = re.sub(r'[^\d.,\-]', '', str(text))
        clean_text = clean_text.replace(',', '')
        
        # Additional validation - ensure we have actual digits
        if not re.search(r'\d', clean_text):
            return None
            
        # Ensure we don't have just dots or dashes
        if clean_text in ['.', '-', '..', '--', '.-', '-.']:
            return None
            
        try:
            value = Decimal(clean_text)
            
            # Handle negative values in parentheses (accounting format)
            if '(' in str(text) and ')' in str(text):
                value = -abs(value)
                
            # Detect scale (millions, billions)
            original_text = str(text).lower()
            if 'billion' in original_text:
                value = value * 1_000_000_000
            elif 'million' in original_text:
                value = value * 1_000_000
            elif 'thousand' in original_text:
                value = value * 1_000
                
            return value
            
        except (ValueError, TypeError, decimal.ConversionSyntax):
            return None

    def _extract_period_from_text(self, text: str) -> str:
        """Extract reporting period from text"""
        # Look for common period patterns
        patterns = [
            r'fiscal\s+year\s+(\d{4})',
            r'year\s+ended\s+\w+\s+\d+,?\s+(\d{4})',
            r'quarter\s+ended\s+\w+\s+\d+,?\s+(\d{4})',
            r'(\d{4})\s+annual\s+report',
            r'q([1-4])\s+(\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._standardize_period(match.group(0))
                
        return "unknown_period"

    def _extract_period_from_table(self, table: List[List[str]]) -> str:
        """Extract period from table headers"""
        if not table:
            return "unknown_period"
            
        header_text = ' '.join([cell or '' for cell in table[0]])
        return self._extract_period_from_text(header_text)

    def _standardize_period(self, period_text: str) -> str:
        """Standardize period format"""
        period_text = period_text.lower().strip()
        
        # Extract year
        year_match = re.search(r'(\d{4})', period_text)
        if not year_match:
            return "unknown_period"
            
        year = year_match.group(1)
        
        # Determine if quarterly or annual
        if re.search(r'q[1-4]|quarter', period_text):
            quarter_match = re.search(r'q([1-4])', period_text)
            if quarter_match:
                return f"Q{quarter_match.group(1)}_{year}"
            else:
                return f"Q_unknown_{year}"
        else:
            return f"FY_{year}"

    def _standardize_metric_name(self, raw_name: str) -> Optional[str]:
        """Map raw metric names to standardized names"""
        raw_name = raw_name.lower().strip()
        
        metric_mappings = {
            'revenue': ['revenue', 'revenues', 'net sales', 'total revenue', 'net revenue'],
            'net_income': ['net income', 'net earnings', 'earnings', 'profit', 'net profit'],
            'total_assets': ['total assets', 'assets'],
            'total_liabilities': ['total liabilities', 'liabilities'],
            'shareholders_equity': ['shareholders equity', 'stockholders equity', 'equity'],
            'operating_income': ['operating income', 'operating profit', 'operating earnings'],
            'rd_expense': ['research and development', 'r&d', 'research development']
        }
        
        for standard_name, variations in metric_mappings.items():
            if any(variation in raw_name for variation in variations):
                return standard_name
                
        return None

    def validate_metrics(self, metrics_dict: Dict[str, float]) -> Dict[str, bool]:
        """Validate extracted metrics for consistency"""
        validation_results = {}
        
        # Revenue should be positive
        if 'revenue' in metrics_dict:
            validation_results['revenue_positive'] = metrics_dict['revenue'] > 0
            
        # Net income should be less than revenue
        if 'revenue' in metrics_dict and 'net_income' in metrics_dict:
            validation_results['income_less_than_revenue'] = (
                metrics_dict['net_income'] <= metrics_dict['revenue']
            )
            
        # Assets should equal liabilities + equity (basic accounting equation)
        if all(k in metrics_dict for k in ['total_assets', 'total_liabilities', 'shareholders_equity']):
            calculated_assets = metrics_dict['total_liabilities'] + metrics_dict['shareholders_equity']
            # Allow 5% tolerance for rounding
            tolerance = metrics_dict['total_assets'] * 0.05
            validation_results['balance_sheet_balances'] = (
                abs(metrics_dict['total_assets'] - calculated_assets) < tolerance
            )
            
        return validation_results

    def _extract_from_xbrl_text(self, text: str, company_id: str, source_doc: str) -> List[FinancialMetric]:
        """
        Extract financial metrics from XBRL content embedded in text.
        """
        from bs4 import BeautifulSoup
        import re
        
        metrics = []
        soup = BeautifulSoup(text, 'lxml-xml')  # Use XML parser for XBRL
        
        # Try case-sensitive search for both tag casings
        tags = soup.find_all(['ix:nonNumeric', 'ix:nonFraction'])
        logger.info(f"Case-sensitive: Found {len(tags)} XBRL tags (ix:nonNumeric or ix:nonFraction)")
        
        # Fallback: search all tags and filter by tag name lowercased
        if len(tags) == 0:
            all_tags = soup.find_all(True)
            tags = [tag for tag in all_tags if tag.name.lower() in ('ix:nonnumeric', 'ix:nonfraction')]
            logger.info(f"Fallback: Found {len(tags)} XBRL tags (ix:nonnumeric or ix:nonfraction, case-insensitive)")
        
        for i, tag in enumerate(tags[:5]):
            logger.info(f"Tag {i+1}: {str(tag)[:200]}")
            logger.info(f"Attributes: {tag.attrs}")
        
        for tag in tags:
            try:
                name = tag.get('name', '')
                if not name.startswith('us-gaap:'):
                    logger.debug(f"Skipping non-us-gaap tag: {name}")
                    continue
                value = tag.get('value')
                if value is None:
                    value = tag.text.strip()
                unit = tag.get('unitref', '')
                context_ref = tag.get('contextref', '')
                if not name or not value:
                    continue
                period = context_ref or ""
                metric_name = self._standardize_metric_name(name)
                if not metric_name:
                    logger.debug(f"Skipping non-standard metric name: {name}")
                    continue
                try:
                    numeric_value = Decimal(value.replace(',', ''))
                except Exception as e:
                    logger.debug(f"Could not parse value for {name}: {value} ({e})")
                    continue
                metrics.append(
                    FinancialMetric(
                        company_id=company_id,
                        metric_name=metric_name,
                        value=numeric_value,
                        period=period,
                        unit=unit,
                        source_document=source_doc,
                        extraction_method="xbrl",
                        confidence=0.95,
                        raw_text=str(tag)[:200]
                    )
                )
                logger.debug(f"Extracted XBRL metric: {metric_name} = {numeric_value} {unit}")
            except Exception as e:
                logger.warning(f"Error parsing XBRL tag: {e}")
        logger.info(f"Extracted {len(metrics)} metrics from XBRL content")
        return metrics


def main():
    """Example usage - demonstrates extraction only"""
    extractor = SECDataExtractor()
    
    # Process a sample document
    sample_file = "data/raw_documents/apple/10-K_2023.pdf"
    
    if Path(sample_file).exists():
        metrics = extractor.process_document(sample_file, "apple")
        
        # Create metrics dictionary for validation
        metrics_dict = {metric.metric_name: float(metric.value) for metric in metrics}
        validation = extractor.validate_metrics(metrics_dict)
        print("Validation results:", validation)
        
        # Print extracted metrics
        for metric in metrics:
            print(f"{metric.metric_name}: ${metric.value:,.2f} ({metric.extraction_method}, confidence: {metric.confidence})")
        
        # Note: Saving to database should be handled by sql_manager.py
        print(f"\nExtracted {len(metrics)} metrics. Use sql_manager.py to save to database.")
    else:
        print(f"Sample file {sample_file} not found")


if __name__ == "__main__":
    main()