"""
SQL Database Manager for Financial Knowledge Base
Handles database operations for financial metrics extracted from documents.
Integrates with the existing FinancialSQLStore for metadata and adds financial metrics management.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from decimal import Decimal

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.src.storage.sql_store import FinancialSQLStore, FinancialMetric as SQLFinancialMetric
from knowledge_base.src.ingestion.sec_sql_extractor import FinancialMetric as ExtractedMetric

logger = logging.getLogger(__name__)


class FinancialMetricsManager:
    """Manages financial metrics storage and retrieval with client isolation."""
    
    def __init__(self):
        self.sql_store = FinancialSQLStore()
    
    def save_extracted_metrics(self, metrics: List[ExtractedMetric], document_id: int) -> bool:
        """
        Save extracted financial metrics to the database.
        
        Args:
            metrics: List of extracted financial metrics
            document_id: ID of the document from which metrics were extracted
            
        Returns:
            bool: Success status
        """
        if not metrics:
            logger.warning("No metrics to save")
            return False
        
        session = self.sql_store.get_session()
        try:
            for metric in metrics:
                # Convert extracted metric to SQL model
                sql_metric = self._convert_to_sql_metric(metric, document_id)
                
                # Check if metric already exists (upsert behavior)
                existing = session.query(SQLFinancialMetric).filter(
                    SQLFinancialMetric.client_id == sql_metric.client_id,
                    SQLFinancialMetric.metric_name == sql_metric.metric_name,
                    SQLFinancialMetric.period_end_date == sql_metric.period_end_date,
                    SQLFinancialMetric.document_id == sql_metric.document_id
                ).first()
                
                if existing:
                    # Update existing metric
                    existing.metric_value = sql_metric.metric_value
                    existing.extraction_confidence = sql_metric.extraction_confidence
                    existing.extraction_method = sql_metric.extraction_method
                    existing.context_text = sql_metric.context_text
                    existing.metadata_json = sql_metric.metadata_json
                    logger.debug(f"Updated existing metric: {metric.metric_name}")
                else:
                    # Add new metric
                    session.add(sql_metric)
                    logger.debug(f"Added new metric: {metric.metric_name}")
            
            session.commit()
            logger.info(f"Successfully saved {len(metrics)} financial metrics")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving financial metrics: {e}")
            return False
        finally:
            session.close()
    
    def _convert_to_sql_metric(self, extracted_metric: ExtractedMetric, document_id: int) -> SQLFinancialMetric:
        """Convert extracted metric to SQL model."""
        
        # Parse period information
        period_type, fiscal_year, fiscal_quarter = self._parse_period(extracted_metric.period)
        
        return SQLFinancialMetric(
            client_id=extracted_metric.company_id,
            document_id=document_id,
            metric_name=extracted_metric.metric_name,
            metric_value=float(extracted_metric.value),
            metric_unit=extracted_metric.unit,
            period_type=period_type,
            period_end_date=extracted_metric.period,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            source_section=extracted_metric.source_document,
            extraction_confidence=extracted_metric.confidence,
            extraction_method=extracted_metric.extraction_method,
            context_text=extracted_metric.raw_text,
            metadata_json={
                'source_document': extracted_metric.source_document,
                'extraction_timestamp': datetime.now().isoformat(),
                'original_period_format': extracted_metric.period
            }
        )
    
    def _parse_period(self, period_str: str) -> tuple[str, int, Optional[int]]:
        """Parse period string to extract type, year, and quarter."""
        period_str = period_str.lower().strip()
        
        if period_str.startswith('fy_'):
            # Annual period: FY_2023
            year = int(period_str.split('_')[1])
            return 'annual', year, None
        elif period_str.startswith('q') and '_' in period_str:
            # Quarterly period: Q1_2023
            parts = period_str.split('_')
            quarter = int(parts[0][1])  # Extract number from Q1, Q2, etc.
            year = int(parts[1])
            return 'quarterly', year, quarter
        else:
            # Default to annual if can't parse
            try:
                year = int(''.join(filter(str.isdigit, period_str))[-4:])  # Last 4 digits
                return 'annual', year, None
            except:
                return 'annual', datetime.now().year, None
    
    def get_metrics_for_client(self, client_id: str, 
                             fiscal_year: Optional[int] = None,
                             metric_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieve financial metrics for a specific client.
        
        Args:
            client_id: Client identifier (ticker symbol)
            fiscal_year: Optional year filter
            metric_names: Optional list of specific metrics to retrieve
            
        Returns:
            List of metric dictionaries
        """
        session = self.sql_store.get_session()
        try:
            query = session.query(SQLFinancialMetric).filter(
                SQLFinancialMetric.client_id == client_id
            )
            
            if fiscal_year:
                query = query.filter(SQLFinancialMetric.fiscal_year == fiscal_year)
            
            if metric_names:
                query = query.filter(SQLFinancialMetric.metric_name.in_(metric_names))
            
            metrics = query.all()
            
            return [
                {
                    'metric_name': m.metric_name,
                    'value': m.metric_value,
                    'unit': m.metric_unit,
                    'period_type': m.period_type,
                    'fiscal_year': m.fiscal_year,
                    'fiscal_quarter': m.fiscal_quarter,
                    'confidence': m.extraction_confidence,
                    'method': m.extraction_method,
                    'context': m.context_text,
                    'created_date': m.created_date.isoformat() if m.created_date else None
                }
                for m in metrics
            ]
            
        finally:
            session.close()
    
    def validate_client_metrics(self, client_id: str, fiscal_year: int) -> Dict[str, bool]:
        """
        Validate financial metrics for accounting consistency.
        
        Args:
            client_id: Client identifier
            fiscal_year: Fiscal year to validate
            
        Returns:
            Dictionary of validation results
        """
        metrics = self.get_metrics_for_client(client_id, fiscal_year)
        
        # Convert to dictionary for easier validation
        metrics_dict = {
            m['metric_name']: m['value'] 
            for m in metrics 
            if m['period_type'] == 'annual'  # Use annual metrics for validation
        }
        
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
    
    def get_comparative_metrics(self, client_ids: List[str], 
                              metric_names: List[str],
                              fiscal_year: int) -> Dict[str, Dict[str, float]]:
        """
        Get comparative metrics across multiple clients (for federated analysis).
        
        Args:
            client_ids: List of client identifiers
            metric_names: List of metrics to compare
            fiscal_year: Fiscal year to compare
            
        Returns:
            Dictionary structured as {client_id: {metric_name: value}}
        """
        session = self.sql_store.get_session()
        try:
            query = session.query(SQLFinancialMetric).filter(
                SQLFinancialMetric.client_id.in_(client_ids),
                SQLFinancialMetric.metric_name.in_(metric_names),
                SQLFinancialMetric.fiscal_year == fiscal_year,
                SQLFinancialMetric.period_type == 'annual'  # Use annual for comparisons
            )
            
            metrics = query.all()
            
            # Structure results
            result = {client_id: {} for client_id in client_ids}
            
            for metric in metrics:
                if metric.client_id in result:
                    result[metric.client_id][metric.metric_name] = metric.metric_value
            
            return result
            
        finally:
            session.close()


def main():
    """Example usage of the financial metrics manager."""
    manager = FinancialMetricsManager()
    
    # Example: Get metrics for a client
    apple_metrics = manager.get_metrics_for_client('AAPL', fiscal_year=2023)
    print(f"Found {len(apple_metrics)} metrics for AAPL in 2023")
    
    # Example: Validate metrics
    if apple_metrics:
        validation = manager.validate_client_metrics('AAPL', 2023)
        print("Validation results:", validation)
    
    # Example: Comparative analysis
    comparative = manager.get_comparative_metrics(
        ['AAPL', 'MSFT', 'GOOGL'], 
        ['revenue', 'net_income'], 
        2023
    )
    print("Comparative metrics:", comparative)


if __name__ == "__main__":
    main()