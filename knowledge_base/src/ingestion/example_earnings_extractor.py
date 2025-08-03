"""
Example Earnings Report Extractor
Demonstrates how to extend the base extractor for different document types.
This is a template for creating extractors for earnings calls, analyst reports, etc.
"""

import re
from typing import Dict, List, Optional
from pathlib import Path
from decimal import Decimal

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.src.ingestion.base_extractor import BaseFinancialExtractor, FinancialMetric


class EarningsReportExtractor(BaseFinancialExtractor):
    """
    Example extractor for earnings reports and analyst documents.
    Demonstrates the extensible pattern for different document types.
    """
    
    def _get_supported_extensions(self) -> List[str]:
        """Earnings reports typically come as PDF, HTML, or text."""
        return ['.pdf', '.html', '.htm', '.txt', '.docx']
    
    def _get_metric_mappings(self) -> Dict[str, str]:
        """Earnings-specific metric mappings."""
        return {
            'revenue': 'revenue',
            'net_income': 'net_income',
            'eps': 'earnings_per_share',
            'ebitda': 'ebitda',
            'free_cash_flow': 'free_cash_flow',
            'operating_margin': 'operating_margin',
            'gross_margin': 'gross_margin'
        }
    
    def _get_financial_patterns(self) -> Dict[str, List[str]]:
        """Patterns specific to earnings reports."""
        return {
            'revenue': [
                r'revenue\s+of\s+\$?\s*([\d,\.]+)\s*(?:million|billion|thousand)?',
                r'total\s+revenue[:\s]+\$?\s*([\d,\.]+)\s*(?:million|billion|thousand)?',
                r'net\s+sales[:\s]+\$?\s*([\d,\.]+)\s*(?:million|billion|thousand)?'
            ],
            'earnings_per_share': [
                r'earnings?\s+per\s+share\s+of\s+\$?\s*([\d,\.\-]+)',
                r'eps\s+of\s+\$?\s*([\d,\.\-]+)',
                r'diluted\s+eps[:\s]+\$?\s*([\d,\.\-]+)'
            ],
            'ebitda': [
                r'ebitda\s+of\s+\$?\s*([\d,\.]+)\s*(?:million|billion|thousand)?',
                r'adjusted\s+ebitda[:\s]+\$?\s*([\d,\.]+)\s*(?:million|billion|thousand)?'
            ],
            'operating_margin': [
                r'operating\s+margin\s+of\s+([\d,\.]+)%',
                r'operating\s+margin[:\s]+([\d,\.]+)%'
            ]
        }
    
    def _additional_validation(self, file_path: Path) -> bool:
        """Additional validation for earnings documents."""
        try:
            # Check if document likely contains earnings information
            if file_path.suffix.lower() == '.pdf':
                # Could extract first page and look for earnings keywords
                return self._check_for_earnings_keywords(file_path)
            return True
        except:
            return False
    
    def _check_for_earnings_keywords(self, file_path: Path) -> bool:
        """Check if document contains earnings-related keywords."""
        earnings_keywords = [
            'earnings', 'quarterly results', 'financial results',
            'q1 ', 'q2 ', 'q3 ', 'q4 ', 'fiscal year', 'revenue', 'eps'
        ]
        
        try:
            # Simple text extraction for validation
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text().lower()
                    return any(keyword in first_page_text for keyword in earnings_keywords)
        except:
            pass
        
        return True  # Default to true if can't validate
    
    def process_document(self, file_path: str, company_id: str, **kwargs) -> List[FinancialMetric]:
        """
        Process earnings report document.
        
        Args:
            file_path: Path to earnings document
            company_id: Company identifier
            **kwargs: Additional parameters like 'quarter', 'fiscal_year'
        """
        file_path = Path(file_path)
        
        if not self.validate_document(file_path):
            return []
        
        print(f"Processing earnings report {file_path.name} for {company_id}")
        
        metrics = []
        
        # Extract text content
        text_content = self._extract_text_content(file_path)
        
        if text_content:
            # Extract period from kwargs or text
            period = kwargs.get('period') or self._extract_period_from_text(text_content)
            
            # Apply financial patterns
            metrics = self._extract_metrics_from_text(text_content, company_id, file_path.name, period)
        
        print(f"Extracted {len(metrics)} metrics from earnings report")
        return metrics
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract text content from different file types."""
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_path.suffix.lower() in ['.html', '.htm']:
                return self._extract_from_html(file_path)
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            # Add more file types as needed
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
        
        return ""
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF."""
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join([page.extract_text() or '' for page in pdf.pages])
        except:
            return ""
    
    def _extract_from_html(self, file_path: Path) -> str:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        except:
            return ""
    
    def _extract_metrics_from_text(self, text: str, company_id: str, 
                                 source_doc: str, period: str) -> List[FinancialMetric]:
        """Extract financial metrics using regex patterns."""
        metrics = []
        
        for metric_name, patterns in self.financial_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    try:
                        raw_value = match.group(1)
                        numeric_value = self._extract_numeric_value(raw_value)
                        
                        if numeric_value is not None:
                            # Determine unit and confidence based on metric type
                            unit = self._determine_unit(metric_name, match.group(0))
                            confidence = self._calculate_confidence(metric_name, match.group(0))
                            
                            metric = FinancialMetric(
                                company_id=company_id,
                                metric_name=metric_name,
                                value=numeric_value,
                                period=period,
                                unit=unit,
                                source_document=source_doc,
                                extraction_method='earnings_regex',
                                confidence=confidence,
                                raw_text=match.group(0),
                                metadata={
                                    'document_type': 'earnings_report',
                                    'pattern_used': pattern
                                }
                            )
                            metrics.append(metric)
                            break  # Take first match for each metric
                    except (ValueError, IndexError):
                        continue
        
        return metrics
    
    def _determine_unit(self, metric_name: str, match_text: str) -> str:
        """Determine the unit for a metric based on context."""
        if metric_name in ['operating_margin', 'gross_margin']:
            return 'percentage'
        elif 'billion' in match_text.lower():
            return 'USD_billions'
        elif 'million' in match_text.lower():
            return 'USD_millions'
        elif 'thousand' in match_text.lower():
            return 'USD_thousands'
        else:
            return 'USD'
    
    def _calculate_confidence(self, metric_name: str, match_text: str) -> float:
        """Calculate confidence score based on match quality."""
        base_confidence = 0.80  # Base confidence for earnings reports
        
        # Increase confidence for more specific patterns
        if metric_name.lower() in match_text.lower():
            base_confidence += 0.05
        
        # Increase confidence if currency symbol is present
        if '$' in match_text:
            base_confidence += 0.05
        
        # Increase confidence for specific keywords
        high_confidence_keywords = ['reported', 'announced', 'generated']
        if any(keyword in match_text.lower() for keyword in high_confidence_keywords):
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)  # Cap at 95%


def main():
    """Example usage of the earnings extractor."""
    extractor = EarningsReportExtractor()
    
    # Register with the global registry
    from knowledge_base.src.ingestion.base_extractor import extractor_registry
    extractor_registry.register('earnings_report', extractor)
    
    print("Earnings Report Extractor registered successfully!")
    print(f"Supported extensions: {extractor._get_supported_extensions()}")
    print(f"Available extractors: {extractor_registry.list_extractors()}")


if __name__ == "__main__":
    main()