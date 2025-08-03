"""
Report generation system for financial analysis and insights.
Creates comprehensive, executive, and comparative reports from financial data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import json
from datetime import datetime
import pandas as pd

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config.settings import get_settings
from knowledge_base.src.retrieval.rag_engine import FinancialRAGEngine
from knowledge_base.src.storage.sql_store import FinancialSQLStore

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    client_id: Optional[str] = None
    report_type: str = "comprehensive"  # comprehensive, executive, comparative, trend
    include_charts: bool = True
    include_tables: bool = True
    output_format: str = "markdown"  # markdown, html, pdf
    template_name: str = "default"
    custom_sections: Optional[List[str]] = None


class FinancialReportGenerator:
    """
    Generates professional financial reports using RAG engine and templates.
    Supports multiple report types and output formats.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.rag_engine = FinancialRAGEngine()
        self.sql_store = FinancialSQLStore()
        
        # Report templates
        self.templates = self._load_templates()
        
        # Output directory
        self.output_dir = Path(self.settings.processing.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_templates(self) -> Dict[str, Dict]:
        """Load report templates."""
        return {
            "comprehensive": {
                "sections": [
                    "executive_summary",
                    "company_overview", 
                    "financial_performance",
                    "risk_analysis",
                    "market_position",
                    "recommendations"
                ],
                "queries": {
                    "executive_summary": "Provide an executive summary of {client_id}'s recent financial performance and key highlights",
                    "company_overview": "Describe {client_id}'s business model, primary operations, and market position",
                    "financial_performance": "Analyze {client_id}'s revenue, profitability, and key financial metrics over recent periods",
                    "risk_analysis": "Identify and analyze the key risk factors for {client_id}",
                    "market_position": "Assess {client_id}'s competitive position and market dynamics",
                    "recommendations": "Based on the analysis, what are the key investment considerations for {client_id}?"
                }
            },
            "executive": {
                "sections": [
                    "key_highlights",
                    "financial_summary",
                    "strategic_recommendations"
                ],
                "queries": {
                    "key_highlights": "What are the most important financial highlights for {client_id}?",
                    "financial_summary": "Summarize {client_id}'s financial position in key bullet points",
                    "strategic_recommendations": "What are the top 3 strategic recommendations for {client_id}?"
                }
            },
            "comparative": {
                "sections": [
                    "comparative_overview",
                    "financial_comparison",
                    "competitive_analysis",
                    "investment_ranking"
                ],
                "queries": {
                    "comparative_overview": "Compare the business models and market positions of {clients}",
                    "financial_comparison": "Compare the financial performance metrics of {clients}",
                    "competitive_analysis": "Analyze the competitive strengths and weaknesses of {clients}",
                    "investment_ranking": "Rank {clients} from an investment perspective with rationale"
                }
            },
            "trend": {
                "sections": [
                    "trend_overview",
                    "historical_analysis",
                    "growth_patterns",
                    "future_outlook"
                ],
                "queries": {
                    "trend_overview": "What are the key financial trends for {client_id} over recent years?",
                    "historical_analysis": "Analyze {client_id}'s historical financial performance and patterns",
                    "growth_patterns": "Identify growth patterns and cyclical trends for {client_id}",
                    "future_outlook": "Based on recent trends, what is the outlook for {client_id}?"
                }
            }
        }
    
    def generate_client_report(
        self,
        client_id: str,
        config: Optional[ReportConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a specific client.
        
        Args:
            client_id: Client identifier
            config: Report configuration
            
        Returns:
            Dictionary with report content and metadata
        """
        if config is None:
            config = ReportConfig(client_id=client_id)
        
        try:
            # Get client information
            client_info = self._get_client_info(client_id)
            if not client_info:
                raise ValueError(f"Client {client_id} not found in database")
            
            # Select template
            template = self.templates.get(config.report_type, self.templates["comprehensive"])
            
            # Generate report sections
            sections = {}
            for section_name in template["sections"]:
                logger.info(f"Generating section: {section_name}")
                
                # Get query template for this section
                query_template = template["queries"].get(section_name, f"Provide information about {section_name} for {client_id}")
                query = query_template.format(client_id=client_id)
                
                # Query RAG engine
                rag_result = self.rag_engine.query(
                    question=query,
                    client_id=client_id,
                    query_type="executive" if "summary" in section_name else "standard",
                    k=8
                )
                
                sections[section_name] = {
                    "content": rag_result["answer"],
                    "sources": rag_result["sources"],
                    "metadata": rag_result["query_metadata"]
                }
            
            # Add financial data tables
            financial_data = None
            if config.include_tables:
                financial_data = self._get_financial_data(client_id)
            
            # Generate charts data
            charts_data = None
            if config.include_charts:
                charts_data = self._generate_charts_data(client_id)
            
            # Compile report
            report = {
                "report_metadata": {
                    "client_id": client_id,
                    "client_info": client_info,
                    "report_type": config.report_type,
                    "generated_date": datetime.now().isoformat(),
                    "template_used": config.template_name,
                    "total_sections": len(sections)
                },
                "sections": sections,
                "financial_data": financial_data,
                "charts_data": charts_data
            }
            
            # Save report
            output_file = self._save_report(report, config)
            report["output_file"] = output_file
            
            logger.info(f"Generated {config.report_type} report for {client_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report for {client_id}: {str(e)}")
            return {
                "error": str(e),
                "client_id": client_id,
                "report_type": config.report_type if config else "unknown"
            }
    
    def generate_comparative_report(
        self,
        client_ids: List[str],
        config: Optional[ReportConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate a comparative analysis report across multiple clients.
        
        Args:
            client_ids: List of client identifiers
            config: Report configuration
            
        Returns:
            Dictionary with comparative report content
        """
        if config is None:
            config = ReportConfig(report_type="comparative")
        
        try:
            # Get client information for all clients
            clients_info = {}
            for client_id in client_ids:
                client_info = self._get_client_info(client_id)
                if client_info:
                    clients_info[client_id] = client_info
            
            if not clients_info:
                raise ValueError("No valid clients found")
            
            # Select template
            template = self.templates["comparative"]
            clients_str = ", ".join(client_ids)
            
            # Generate comparative sections
            sections = {}
            for section_name in template["sections"]:
                logger.info(f"Generating comparative section: {section_name}")
                
                query_template = template["queries"][section_name]
                query = query_template.format(clients=clients_str)
                
                # Cross-client query
                rag_result = self.rag_engine.query(
                    question=query,
                    query_type="comparative",
                    enable_cross_client=True,
                    filters={"client_ids": client_ids},
                    k=12
                )
                
                sections[section_name] = {
                    "content": rag_result["answer"],
                    "sources": rag_result["sources"],
                    "metadata": rag_result["query_metadata"]
                }
            
            # Add comparative financial data
            comparative_data = None
            if config.include_tables:
                comparative_data = self._get_comparative_financial_data(client_ids)
            
            # Generate comparative charts
            comparative_charts = None
            if config.include_charts:
                comparative_charts = self._generate_comparative_charts_data(client_ids)
            
            # Compile report
            report = {
                "report_metadata": {
                    "client_ids": client_ids,
                    "clients_info": clients_info,
                    "report_type": "comparative",
                    "generated_date": datetime.now().isoformat(),
                    "total_sections": len(sections)
                },
                "sections": sections,
                "comparative_data": comparative_data,
                "comparative_charts": comparative_charts
            }
            
            # Save report
            output_file = self._save_report(report, config)
            report["output_file"] = output_file
            
            logger.info(f"Generated comparative report for {', '.join(client_ids)}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating comparative report: {str(e)}")
            return {
                "error": str(e),
                "client_ids": client_ids,
                "report_type": "comparative"
            }
    
    def _get_client_info(self, client_id: str) -> Optional[Dict]:
        """Get client information from SQL database."""
        session = self.sql_store.get_session()
        try:
            from knowledge_base.src.storage.sql_store import Client
            
            client = session.query(Client).filter(Client.id == client_id).first()
            if client:
                return {
                    "id": client.id,
                    "company_name": client.company_name,
                    "industry": client.industry,
                    "sector": client.sector,
                    "cik": client.cik,
                    "market_cap": client.market_cap
                }
            return None
        finally:
            session.close()
    
    def _get_financial_data(self, client_id: str) -> Optional[Dict]:
        """Get financial data for tables."""
        try:
            # Get recent financial metrics
            metrics_df = self.sql_store.get_financial_metrics(client_id=client_id)
            
            if metrics_df.empty:
                return None
            
            # Organize by metric type and period
            financial_summary = {}
            
            # Group by metric name
            for metric_name in metrics_df['metric_name'].unique():
                metric_data = metrics_df[metrics_df['metric_name'] == metric_name].copy()
                metric_data = metric_data.sort_values('fiscal_year', ascending=False)
                
                financial_summary[metric_name] = metric_data.to_dict('records')
            
            return {
                "summary_table": financial_summary,
                "latest_metrics": metrics_df.head(10).to_dict('records'),
                "data_points": len(metrics_df)
            }
            
        except Exception as e:
            logger.warning(f"Error getting financial data for {client_id}: {str(e)}")
            return None
    
    def _get_comparative_financial_data(self, client_ids: List[str]) -> Optional[Dict]:
        """Get comparative financial data across clients."""
        try:
            all_metrics = []
            
            for client_id in client_ids:
                client_metrics = self.sql_store.get_financial_metrics(client_id=client_id)
                if not client_metrics.empty:
                    all_metrics.append(client_metrics)
            
            if not all_metrics:
                return None
            
            # Combine all metrics
            combined_df = pd.concat(all_metrics, ignore_index=True)
            
            # Create comparison tables
            comparison_data = {}
            
            # Latest metrics by client
            latest_by_client = {}
            for client_id in client_ids:
                client_data = combined_df[combined_df['client_id'] == client_id]
                if not client_data.empty:
                    # Get most recent year for each metric
                    latest_metrics = client_data.loc[
                        client_data.groupby('metric_name')['fiscal_year'].idxmax()
                    ]
                    latest_by_client[client_id] = latest_metrics.to_dict('records')
            
            comparison_data["latest_by_client"] = latest_by_client
            
            # Pivot table for side-by-side comparison
            pivot_data = combined_df.pivot_table(
                index=['metric_name', 'fiscal_year'],
                columns='client_id',
                values='metric_value',
                aggfunc='first'
            ).reset_index()
            
            comparison_data["pivot_table"] = pivot_data.to_dict('records')
            
            return comparison_data
            
        except Exception as e:
            logger.warning(f"Error getting comparative financial data: {str(e)}")
            return None
    
    def _generate_charts_data(self, client_id: str) -> Optional[Dict]:
        """Generate data for charts and visualizations."""
        try:
            metrics_df = self.sql_store.get_financial_metrics(client_id=client_id)
            
            if metrics_df.empty:
                return None
            
            charts_data = {}
            
            # Revenue trend chart
            revenue_data = metrics_df[metrics_df['metric_name'] == 'revenue'].copy()
            if not revenue_data.empty:
                revenue_data = revenue_data.sort_values('fiscal_year')
                charts_data["revenue_trend"] = {
                    "type": "line",
                    "data": {
                        "x": revenue_data['fiscal_year'].tolist(),
                        "y": revenue_data['metric_value'].tolist(),
                        "title": f"{client_id} Revenue Trend"
                    }
                }
            
            # Profitability metrics
            profit_metrics = ['net_income', 'gross_profit', 'operating_income']
            profit_data = metrics_df[metrics_df['metric_name'].isin(profit_metrics)].copy()
            if not profit_data.empty:
                latest_year = profit_data['fiscal_year'].max()
                latest_profits = profit_data[profit_data['fiscal_year'] == latest_year]
                
                charts_data["profitability"] = {
                    "type": "bar",
                    "data": {
                        "labels": latest_profits['metric_name'].tolist(),
                        "values": latest_profits['metric_value'].tolist(),
                        "title": f"{client_id} Profitability Metrics ({latest_year})"
                    }
                }
            
            return charts_data
            
        except Exception as e:
            logger.warning(f"Error generating charts data for {client_id}: {str(e)}")
            return None
    
    def _generate_comparative_charts_data(self, client_ids: List[str]) -> Optional[Dict]:
        """Generate comparative charts data."""
        try:
            charts_data = {}
            
            # Get latest revenue for each client
            revenue_comparison = {}
            for client_id in client_ids:
                metrics_df = self.sql_store.get_financial_metrics(
                    client_id=client_id,
                    metric_names=['revenue']
                )
                if not metrics_df.empty:
                    latest_revenue = metrics_df.loc[metrics_df['fiscal_year'].idxmax()]
                    revenue_comparison[client_id] = latest_revenue['metric_value']
            
            if revenue_comparison:
                charts_data["revenue_comparison"] = {
                    "type": "bar",
                    "data": {
                        "labels": list(revenue_comparison.keys()),
                        "values": list(revenue_comparison.values()),
                        "title": "Revenue Comparison"
                    }
                }
            
            return charts_data
            
        except Exception as e:
            logger.warning(f"Error generating comparative charts: {str(e)}")
            return None
    
    def _save_report(self, report: Dict, config: ReportConfig) -> str:
        """Save report to file in specified format."""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if config.client_id:
                filename = f"{config.client_id}_{config.report_type}_{timestamp}"
            else:
                filename = f"comparative_{config.report_type}_{timestamp}"
            
            if config.output_format == "markdown":
                return self._save_markdown_report(report, filename)
            elif config.output_format == "html":
                return self._save_html_report(report, filename)
            elif config.output_format == "json":
                return self._save_json_report(report, filename)
            else:
                # Default to markdown
                return self._save_markdown_report(report, filename)
                
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return ""
    
    def _save_markdown_report(self, report: Dict, filename: str) -> str:
        """Save report as markdown file."""
        filepath = self.output_dir / f"{filename}.md"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Header
                metadata = report.get("report_metadata", {})
                f.write(f"# Financial Analysis Report\n\n")
                f.write(f"**Generated:** {metadata.get('generated_date', 'Unknown')}\n")
                f.write(f"**Report Type:** {metadata.get('report_type', 'Unknown')}\n")
                
                if 'client_id' in metadata:
                    client_info = metadata.get('client_info', {})
                    f.write(f"**Company:** {client_info.get('company_name', metadata['client_id'])}\n")
                    f.write(f"**Industry:** {client_info.get('industry', 'Unknown')}\n")
                elif 'client_ids' in metadata:
                    f.write(f"**Companies:** {', '.join(metadata['client_ids'])}\n")
                
                f.write("\n---\n\n")
                
                # Sections
                sections = report.get("sections", {})
                for section_name, section_data in sections.items():
                    f.write(f"## {section_name.replace('_', ' ').title()}\n\n")
                    f.write(f"{section_data['content']}\n\n")
                    
                    # Add sources if available
                    sources = section_data.get('sources', [])
                    if sources:
                        f.write("### Sources\n\n")
                        for i, source in enumerate(sources[:3]):  # Limit to top 3 sources
                            metadata_str = source.get('metadata', {})
                            client_id = metadata_str.get('client_id', 'Unknown')
                            filing_type = metadata_str.get('filing_type', 'Unknown')
                            f.write(f"{i+1}. {client_id} {filing_type} filing\n")
                        f.write("\n")
                
                # Financial data tables
                if report.get("financial_data"):
                    f.write("## Financial Data\n\n")
                    f.write("*[Financial tables would be rendered here]*\n\n")
                
                # Charts
                if report.get("charts_data"):
                    f.write("## Charts and Visualizations\n\n")
                    f.write("*[Charts would be rendered here]*\n\n")
            
            logger.info(f"Saved markdown report: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving markdown report: {str(e)}")
            return ""
    
    def _save_json_report(self, report: Dict, filename: str) -> str:
        """Save report as JSON file for programmatic access."""
        filepath = self.output_dir / f"{filename}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Saved JSON report: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving JSON report: {str(e)}")
            return ""
    
    def _save_html_report(self, report: Dict, filename: str) -> str:
        """Save report as HTML file."""
        # For MVP, just save as JSON and return the path
        # In a full implementation, this would generate proper HTML
        return self._save_json_report(report, filename)


def main():
    """Example usage of report generator."""
    generator = FinancialReportGenerator()
    
    # Generate client report
    print("Generating client report for AAPL...")
    config = ReportConfig(
        client_id="AAPL",
        report_type="comprehensive",
        output_format="markdown"
    )
    
    report = generator.generate_client_report("AAPL", config)
    
    if "error" not in report:
        print(f"Report generated successfully: {report.get('output_file')}")
        print(f"Sections: {list(report['sections'].keys())}")
    else:
        print(f"Error: {report['error']}")
    
    # Generate comparative report
    print("\nGenerating comparative report...")
    comparative_config = ReportConfig(
        report_type="comparative",
        output_format="markdown"
    )
    
    comparative_report = generator.generate_comparative_report(
        ["AAPL", "MSFT", "GOOGL"],
        comparative_config
    )
    
    if "error" not in comparative_report:
        print(f"Comparative report generated: {comparative_report.get('output_file')}")
    else:
        print(f"Error: {comparative_report['error']}")


if __name__ == "__main__":
    main()