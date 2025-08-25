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
Regulatory Compliance: XBRL is the official structured format.


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

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.src.ingestion.base_extractor import BaseFinancialExtractor, FinancialMetric

class SECDataExtractor(BaseFinancialExtractor):
    """Extract structured financial data from SEC filings"""
    
    def __init__(self):
        super().__init__()
        # Removed database initialization - handled by sql_manager
    
    def _get_supported_extensions(self) -> List[str]:
        """SEC documents support PDF, XML, HTML, and text files."""
        return ['.pdf', '.xml', '.xbrl', '.html', '.htm', '.txt']
    
    def _get_metric_mappings(self) -> Dict[str, str]:
        """Standard GAAP metric mappings for SEC documents."""
        return {
            'us-gaap:Revenues': 'revenue',
            'us-gaap:NetIncomeLoss': 'net_income',
            'us-gaap:Assets': 'total_assets',
            'us-gaap:Liabilities': 'total_liabilities',
            'us-gaap:StockholdersEquity': 'shareholders_equity',
            'us-gaap:CashAndCashEquivalentsAtCarryingValue': 'cash_equivalents',
            'us-gaap:ResearchAndDevelopmentExpense': 'rd_expense',
            'us-gaap:OperatingIncomeLoss': 'operating_income'
        }
    
    def _get_financial_patterns(self) -> Dict[str, List[str]]:
        """Common financial statement patterns for SEC PDF fallback."""
        return {
            'revenue': [
                r'(?:net\s+)?(?:revenues?|sales|net\s+sales)\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'total\s+(?:net\s+)?revenues?\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?'
            ],
            'net_income': [
                r'net\s+(?:income|earnings?)\s*[:\s]*\$?\s*([\d,\-]+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                r'(?:net\s+)?(?:income|earnings?)\s+(?:loss\s+)?attributable.*?\$?\s*([\d,\-]+(?:\.\d+)?)\s*(?:million|billion|thousand)?'
            ],
            'total_assets': [
                r'total\s+assets\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?'
            ]
        }
    
    def _additional_validation(self, file_path: Path) -> bool:
        """Additional validation for SEC documents."""
        # Check if it's likely an SEC document by looking for common identifiers
        try:
            if file_path.suffix.lower() in ['.pdf', '.html', '.htm']:
                # Could add more sophisticated validation here
                # e.g., checking for SEC headers, CIK numbers, etc.
                return True
            elif file_path.suffix.lower() in ['.xml', '.xbrl']:
                # Validate XBRL structure
                return self._validate_xbrl_file(file_path)
            return True
        except:
            return False
    
    def _validate_xbrl_file(self, file_path: Path) -> bool:
        """Validate if XML file is proper XBRL."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            # Check for XBRL elements in a portable way
            return any('xbrl' in elem.tag.lower() for elem in root.iter())
        except:
            return False

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
            
            for gaap_tag, metric_name in self.metric_mappings.items():
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
                
            return self._extract_from_text_content(text, company_id, file_path.name)
            
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return []

    def _extract_from_text_content(self, text: str, company_id: str, source_doc: str) -> List[FinancialMetric]:
        """Extract metrics from text content using patterns"""
        metrics = []
        period = self._extract_period_from_text(text)
        
        for metric_name, patterns in self.financial_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
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
                                extraction_method='text_regex',
                                confidence=0.70,
                                raw_text=match.group(0)
                            )
                            metrics.append(metric)
                            break
                            
                    except (ValueError, IndexError):
                        continue
                        
        return metrics

    # _extract_numeric_value inherited from base class

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

    # _standardize_period inherited from base class

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

    # validate_metrics inherited from base class


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