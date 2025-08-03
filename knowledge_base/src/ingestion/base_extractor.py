"""
Base Abstract Extractor for Financial Documents
Provides a common interface for extracting structured financial data from various document types.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal


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
    metadata: Optional[Dict[str, Any]] = None


class BaseFinancialExtractor(ABC):
    """
    Abstract base class for financial document extractors.
    
    This provides a common interface for extracting structured financial metrics
    from different document types (SEC filings, earnings reports, annual reports, etc.).
    """
    
    def __init__(self):
        """Initialize the extractor with document-specific configurations."""
        self.supported_extensions = self._get_supported_extensions()
        self.metric_mappings = self._get_metric_mappings()
        self.financial_patterns = self._get_financial_patterns()
    
    @abstractmethod
    def _get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions (e.g., ['.pdf', '.xml', '.html'])."""
        pass
    
    @abstractmethod
    def _get_metric_mappings(self) -> Dict[str, str]:
        """Return document-specific metric name mappings."""
        pass
    
    @abstractmethod
    def _get_financial_patterns(self) -> Dict[str, List[str]]:
        """Return document-specific regex patterns for financial metrics."""
        pass
    
    @abstractmethod
    def process_document(self, file_path: str, company_id: str, **kwargs) -> List[FinancialMetric]:
        """
        Main entry point for processing a document.
        
        Args:
            file_path: Path to the document file
            company_id: Identifier for the company (ticker symbol, etc.)
            **kwargs: Additional document-specific parameters
            
        Returns:
            List of extracted financial metrics
        """
        pass
    
    def validate_document(self, file_path: str) -> bool:
        """
        Validate if the document can be processed by this extractor.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if document can be processed
        """
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            return False
        
        # Check file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Subclasses can override for additional validation
        return self._additional_validation(file_path)
    
    def _additional_validation(self, file_path: Path) -> bool:
        """Override for document-specific validation logic."""
        return True
    
    def _extract_numeric_value(self, text: str) -> Optional[Decimal]:
        """Standard numeric value extraction (can be overridden)."""
        if not text:
            return None
            
        # Remove common formatting
        import re
        clean_text = re.sub(r'[^\d.,\-]', '', str(text))
        clean_text = clean_text.replace(',', '')
        
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
            
        except (ValueError, TypeError):
            return None
    
    def _standardize_period(self, period_text: str) -> str:
        """Standard period format standardization (can be overridden)."""
        import re
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
    
    def validate_metrics(self, metrics_dict: Dict[str, float]) -> Dict[str, bool]:
        """
        Standard financial metrics validation.
        
        Args:
            metrics_dict: Dictionary of metric_name -> value
            
        Returns:
            Dictionary of validation results
        """
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
            tolerance = abs(metrics_dict['total_assets'] * 0.05)
            validation_results['balance_sheet_balances'] = (
                abs(metrics_dict['total_assets'] - calculated_assets) <= tolerance
            )
        
        return validation_results


class ExtractorRegistry:
    """Registry for managing multiple document extractors."""
    
    def __init__(self):
        self._extractors: Dict[str, BaseFinancialExtractor] = {}
    
    def register(self, name: str, extractor: BaseFinancialExtractor):
        """Register an extractor with a name."""
        self._extractors[name] = extractor
    
    def get_extractor(self, name: str) -> Optional[BaseFinancialExtractor]:
        """Get an extractor by name."""
        return self._extractors.get(name)
    
    def get_extractor_for_file(self, file_path: str) -> Optional[BaseFinancialExtractor]:
        """Find the best extractor for a given file."""
        file_path = Path(file_path)
        
        for extractor in self._extractors.values():
            if extractor.validate_document(file_path):
                return extractor
        
        return None
    
    def list_extractors(self) -> List[str]:
        """List all registered extractor names."""
        return list(self._extractors.keys())


# Global registry instance
extractor_registry = ExtractorRegistry()