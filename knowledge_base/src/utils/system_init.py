"""
System initialization and demo setup utilities.
Handles component initialization and demo data setup.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
from datetime import datetime

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config.settings import get_settings, validate_api_keys
from knowledge_base.src.ingestion.sec_downloader import SECDownloader
from knowledge_base.src.ingestion.document_processor import FinancialDocumentProcessor
from knowledge_base.src.storage.vector_store import FinancialVectorStore
from knowledge_base.src.storage.sql_store import FinancialSQLStore
from knowledge_base.src.retrieval.rag_engine import FinancialRAGEngine
from knowledge_base.src.generation.report_generator import FinancialReportGenerator

logger = logging.getLogger(__name__)


class SystemInitializer:
    """Handles system initialization and setup."""
    
    def __init__(self):
        self.settings = get_settings()
        self.components = {}
    
    def initialize_system(self, force: bool = False) -> Dict[str, Any]:
        """
        Initialize all system components.
        
        Args:
            force: Force reinitialization even if components exist
            
        Returns:
            Dictionary with initialization results
        """
        results = {
            "success": True,
            "components_initialized": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate API keys
            api_status = validate_api_keys()
            if not api_status["openai_available"]:
                results["warnings"].append("OpenAI API not available - system will run in fallback mode")
            
            # Initialize storage components
            logger.info("Initializing storage components...")
            
            # SQL Store
            try:
                self.components["sql_store"] = FinancialSQLStore()
                results["components_initialized"].append("sql_store")
                logger.info("✅ SQL store initialized")
            except Exception as e:
                error_msg = f"Failed to initialize SQL store: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Vector Store
            try:
                self.components["vector_store"] = FinancialVectorStore()
                results["components_initialized"].append("vector_store")
                logger.info("✅ Vector store initialized")
            except Exception as e:
                error_msg = f"Failed to initialize vector store: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Initialize processing components
            logger.info("Initializing processing components...")
            
            # SEC Downloader
            try:
                self.components["downloader"] = SECDownloader()
                results["components_initialized"].append("downloader")
                logger.info("✅ SEC downloader initialized")
            except Exception as e:
                error_msg = f"Failed to initialize SEC downloader: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Document Processor
            try:
                self.components["processor"] = FinancialDocumentProcessor()
                results["components_initialized"].append("processor")
                logger.info("✅ Document processor initialized")
            except Exception as e:
                error_msg = f"Failed to initialize document processor: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Initialize AI components
            logger.info("Initializing AI components...")
            
            # RAG Engine
            try:
                self.components["rag_engine"] = FinancialRAGEngine()
                results["components_initialized"].append("rag_engine")
                logger.info("✅ RAG engine initialized")
            except Exception as e:
                error_msg = f"Failed to initialize RAG engine: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Report Generator
            try:
                self.components["report_generator"] = FinancialReportGenerator()
                results["components_initialized"].append("report_generator")
                logger.info("✅ Report generator initialized")
            except Exception as e:
                error_msg = f"Failed to initialize report generator: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Check if any critical components failed
            critical_components = ["sql_store", "vector_store"]
            missing_critical = [comp for comp in critical_components if comp not in results["components_initialized"]]
            
            if missing_critical:
                results["success"] = False
                results["errors"].append(f"Critical components failed to initialize: {missing_critical}")
            
            logger.info(f"System initialization completed. Success: {results['success']}")
            return results
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"System initialization failed: {str(e)}")
            logger.error(f"System initialization failed: {str(e)}")
            return results
    
    def create_demo_data(self) -> Dict[str, Any]:
        """
        Create demo data for testing and demonstration.
        
        Returns:
            Dictionary with demo data creation results
        """
        results = {
            "success": True,
            "demo_clients_created": [],
            "errors": [],
            "files_created": []
        }
        
        try:
            if "sql_store" not in self.components:
                results["errors"].append("SQL store not initialized")
                results["success"] = False
                return results
            
            sql_store = self.components["sql_store"]
            
            # Demo companies
            demo_companies = [
                {
                    "id": "AAPL",
                    "company_name": "Apple Inc.",
                    "cik": "0000320193",
                    "industry": "Technology",
                    "sector": "Consumer Electronics",
                    "market_cap": 3000000000000  # $3T
                },
                {
                    "id": "MSFT",
                    "company_name": "Microsoft Corporation",
                    "cik": "0000789019",
                    "industry": "Technology",
                    "sector": "Software",
                    "market_cap": 2800000000000  # $2.8T
                },
                {
                    "id": "GOOGL",
                    "company_name": "Alphabet Inc.",
                    "cik": "0001652044",
                    "industry": "Technology",
                    "sector": "Internet Services",
                    "market_cap": 1700000000000  # $1.7T
                }
            ]
            
            # Create demo clients
            for company in demo_companies:
                try:
                    success = sql_store.add_client(company)
                    if success:
                        results["demo_clients_created"].append(company["id"])
                        logger.info(f"✅ Created demo client: {company['id']}")
                    else:
                        results["errors"].append(f"Failed to create client: {company['id']}")
                except Exception as e:
                    results["errors"].append(f"Error creating client {company['id']}: {str(e)}")
            
            # Create demo documents
            demo_documents = self._create_demo_documents()
            
            for doc_data in demo_documents:
                try:
                    doc_id = sql_store.add_document(doc_data)
                    if doc_id:
                        logger.info(f"✅ Created demo document: {doc_data['document_id']}")
                except Exception as e:
                    results["errors"].append(f"Error creating document {doc_data['document_id']}: {str(e)}")
            
            # Create demo financial metrics
            demo_metrics = self._create_demo_metrics()
            
            try:
                success = sql_store.add_financial_metrics(demo_metrics)
                if success:
                    logger.info(f"✅ Created {len(demo_metrics)} demo financial metrics")
                else:
                    results["errors"].append("Failed to create demo financial metrics")
            except Exception as e:
                results["errors"].append(f"Error creating demo metrics: {str(e)}")
            
            if results["errors"]:
                results["success"] = False
            
            return results
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Demo data creation failed: {str(e)}")
            logger.error(f"Demo data creation failed: {str(e)}")
            return results
    
    def _create_demo_documents(self) -> List[Dict]:
        """Create demo document metadata."""
        return [
            {
                "document_id": "AAPL_10K_2023",
                "client_id": "AAPL",
                "filing_type": "10-K",
                "filing_date": "2023-10-27",
                "period_of_report": "2023-09-30",
                "fiscal_year_end": "0930",
                "file_path": "/demo/aapl_10k_2023.txt",
                "file_size": 2500000,
                "download_date": datetime.utcnow(),
                "processed_date": datetime.utcnow(),
                "total_chunks": 45,
                "financial_density": 18.5,
                "has_revenue_data": True,
                "has_profit_data": True,
                "has_balance_sheet": True,
                "has_cash_flow": True,
                "metadata_json": {"source": "demo", "quality": "high"}
            },
            {
                "document_id": "MSFT_10K_2023",
                "client_id": "MSFT",
                "filing_type": "10-K",
                "filing_date": "2023-07-31",
                "period_of_report": "2023-06-30",
                "fiscal_year_end": "0630",
                "file_path": "/demo/msft_10k_2023.txt",
                "file_size": 2200000,
                "download_date": datetime.utcnow(),
                "processed_date": datetime.utcnow(),
                "total_chunks": 42,
                "financial_density": 16.8,
                "has_revenue_data": True,
                "has_profit_data": True,
                "has_balance_sheet": True,
                "has_cash_flow": True,
                "metadata_json": {"source": "demo", "quality": "high"}
            },
            {
                "document_id": "GOOGL_10K_2023",
                "client_id": "GOOGL",
                "filing_type": "10-K",
                "filing_date": "2023-02-02",
                "period_of_report": "2022-12-31",
                "fiscal_year_end": "1231",
                "file_path": "/demo/googl_10k_2023.txt",
                "file_size": 1800000,
                "download_date": datetime.utcnow(),
                "processed_date": datetime.utcnow(),
                "total_chunks": 38,
                "financial_density": 14.2,
                "has_revenue_data": True,
                "has_profit_data": True,
                "has_balance_sheet": True,
                "has_cash_flow": True,
                "metadata_json": {"source": "demo", "quality": "high"}
            }
        ]
    
    def _create_demo_metrics(self) -> List[Dict]:
        """Create demo financial metrics."""
        return [
            # Apple metrics
            {
                "client_id": "AAPL",
                "document_id": 1,
                "metric_name": "revenue",
                "metric_value": 394328000000,  # $394.3B
                "metric_unit": "USD",
                "period_type": "annual",
                "period_end_date": "2023-09-30",
                "fiscal_year": 2023,
                "source_section": "Consolidated Statements of Operations",
                "extraction_confidence": 0.95,
                "extraction_method": "demo"
            },
            {
                "client_id": "AAPL",
                "document_id": 1,
                "metric_name": "net_income",
                "metric_value": 96995000000,  # $97B
                "metric_unit": "USD",
                "period_type": "annual",
                "period_end_date": "2023-09-30",
                "fiscal_year": 2023,
                "source_section": "Consolidated Statements of Operations",
                "extraction_confidence": 0.95,
                "extraction_method": "demo"
            },
            # Microsoft metrics
            {
                "client_id": "MSFT",
                "document_id": 2,
                "metric_name": "revenue",
                "metric_value": 211915000000,  # $211.9B
                "metric_unit": "USD",
                "period_type": "annual",
                "period_end_date": "2023-06-30",
                "fiscal_year": 2023,
                "source_section": "Consolidated Statements of Operations",
                "extraction_confidence": 0.95,
                "extraction_method": "demo"
            },
            {
                "client_id": "MSFT",
                "document_id": 2,
                "metric_name": "net_income",
                "metric_value": 72361000000,  # $72.4B
                "metric_unit": "USD",
                "period_type": "annual",
                "period_end_date": "2023-06-30",
                "fiscal_year": 2023,
                "source_section": "Consolidated Statements of Operations",
                "extraction_confidence": 0.95,
                "extraction_method": "demo"
            },
            # Google metrics
            {
                "client_id": "GOOGL",
                "document_id": 3,
                "metric_name": "revenue",
                "metric_value": 282836000000,  # $282.8B
                "metric_unit": "USD",
                "period_type": "annual",
                "period_end_date": "2022-12-31",
                "fiscal_year": 2022,
                "source_section": "Consolidated Statements of Income",
                "extraction_confidence": 0.95,
                "extraction_method": "demo"
            },
            {
                "client_id": "GOOGL",
                "document_id": 3,
                "metric_name": "net_income",
                "metric_value": 59972000000,  # $60B
                "metric_unit": "USD",
                "period_type": "annual",
                "period_end_date": "2022-12-31",
                "fiscal_year": 2022,
                "source_section": "Consolidated Statements of Income",
                "extraction_confidence": 0.95,
                "extraction_method": "demo"
            }
        ]
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Perform system health check.
        
        Returns:
            Dictionary with health check results
        """
        health = {
            "overall_status": "healthy",
            "components": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check each component
            for component_name, component in self.components.items():
                try:
                    if component_name == "sql_store":
                        # Test SQL store
                        stats = component.get_client_statistics()
                        health["components"]["sql_store"] = {
                            "status": "healthy",
                            "clients": len(stats) if not stats.empty else 0
                        }
                    
                    elif component_name == "vector_store":
                        # Test vector store
                        clients = component.list_clients()
                        health["components"]["vector_store"] = {
                            "status": "healthy",
                            "clients": len(clients)
                        }
                    
                    else:
                        # Basic component check
                        health["components"][component_name] = {"status": "healthy"}
                
                except Exception as e:
                    health["components"][component_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health["issues"].append(f"{component_name}: {str(e)}")
            
            # Check API connectivity
            api_status = validate_api_keys()
            if not api_status["openai_available"]:
                health["issues"].append("OpenAI API not available")
                health["recommendations"].append("Configure OpenAI API key for full functionality")
            
            # Determine overall status
            if health["issues"]:
                if len(health["issues"]) > len(self.components) / 2:
                    health["overall_status"] = "unhealthy"
                else:
                    health["overall_status"] = "degraded"
            
            return health
            
        except Exception as e:
            health["overall_status"] = "error"
            health["issues"].append(f"Health check failed: {str(e)}")
            return health
    
    def get_components(self) -> Dict[str, Any]:
        """Get initialized components."""
        return self.components


def initialize_demo_system() -> Dict[str, Any]:
    """
    Convenience function to initialize system with demo data.
    
    Returns:
        Dictionary with initialization results
    """
    initializer = SystemInitializer()
    
    # Initialize system
    init_results = initializer.initialize_system()
    
    if init_results["success"]:
        # Create demo data
        demo_results = initializer.create_demo_data()
        init_results["demo_data"] = demo_results
        
        # Check system health
        health = initializer.check_system_health()
        init_results["health"] = health
    
    return init_results


def main():
    """Example usage of system initializer."""
    logger.info("Initializing Financial Knowledge Base RAG System...")
    
    results = initialize_demo_system()
    
    if results["success"]:
        print("✅ System initialized successfully!")
        
        if results.get("demo_data", {}).get("success"):
            demo_clients = results["demo_data"]["demo_clients_created"]
            print(f"✅ Demo data created for {len(demo_clients)} clients: {', '.join(demo_clients)}")
        
        health = results.get("health", {})
        print(f"🏥 System health: {health.get('overall_status', 'unknown')}")
        
        if health.get("issues"):
            print("⚠️  Issues found:")
            for issue in health["issues"]:
                print(f"   - {issue}")
        
        if health.get("recommendations"):
            print("💡 Recommendations:")
            for rec in health["recommendations"]:
                print(f"   - {rec}")
    
    else:
        print("❌ System initialization failed!")
        for error in results["errors"]:
            print(f"   - {error}")


if __name__ == "__main__":
    main()