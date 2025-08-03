#!/usr/bin/env python3
"""
Quickstart script for Financial Knowledge Base RAG System.
Demonstrates basic usage and provides interactive examples.
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from knowledge_base.src.utils.system_init import SystemInitializer, initialize_demo_system
from knowledge_base.config.settings import get_settings, validate_api_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        Financial Knowledge Base RAG System - Quick Start     ║
    ║                                                              ║
    ║        🏦 Enterprise AI for Financial Document Analysis      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_prerequisites():
    """Check system prerequisites."""
    print("🔍 Checking prerequisites...")
    
    issues = []
    recommendations = []
    
    # Check Python version
    if sys.version_info < (3, 9):
        issues.append(f"Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    else:
        print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check API keys
    api_status = validate_api_keys()
    if api_status["openai_available"]:
        print("  ✅ OpenAI API key configured")
    else:
        print("  ⚠️  OpenAI API key not found")
        recommendations.append("Set OPENAI_API_KEY environment variable for full functionality")
        recommendations.append("System will run in fallback mode with local models")
    
    # Check required directories
    settings = get_settings()
    required_dirs = [
        Path(settings.processing.raw_data_path),
        Path(settings.processing.processed_data_path),
        Path(settings.processing.output_path),
        Path(settings.database.faiss_index_path).parent
    ]
    
    for directory in required_dirs:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created directory: {directory}")
            except Exception as e:
                issues.append(f"Cannot create directory {directory}: {str(e)}")
    
    return issues, recommendations


def initialize_system():
    """Initialize the system."""
    print("\n🚀 Initializing system components...")
    
    results = initialize_demo_system()
    
    if results["success"]:
        print("  ✅ System initialized successfully!")
        
        # Show component status
        components = results.get("components_initialized", [])
        print(f"  📦 Components initialized: {', '.join(components)}")
        
        # Show demo data status
        demo_data = results.get("demo_data", {})
        if demo_data.get("success"):
            demo_clients = demo_data.get("demo_clients_created", [])
            print(f"  🏢 Demo clients created: {', '.join(demo_clients)}")
        
        # Show health status
        health = results.get("health", {})
        status = health.get("overall_status", "unknown")
        print(f"  🏥 System health: {status}")
        
        if health.get("issues"):
            print("  ⚠️  Issues detected:")
            for issue in health["issues"]:
                print(f"     - {issue}")
        
        return True
    else:
        print("  ❌ System initialization failed!")
        for error in results.get("errors", []):
            print(f"     - {error}")
        return False


def run_demo_queries():
    """Run demonstration queries."""
    print("\n💬 Running demo queries...")
    
    try:
        from financial_kb.src.retrieval.rag_engine import FinancialRAGEngine
        
        rag_engine = FinancialRAGEngine()
        
        demo_queries = [
            {
                "question": "What are the key financial highlights for Apple?",
                "client_id": "AAPL",
                "description": "Client-specific query for Apple"
            },
            {
                "question": "Compare the revenue of Apple, Microsoft, and Google",
                "enable_cross_client": True,
                "description": "Cross-client comparative query"
            },
            {
                "question": "What are the main business segments for Microsoft?",
                "client_id": "MSFT",
                "description": "Business analysis query"
            }
        ]
        
        for i, query_config in enumerate(demo_queries, 1):
            print(f"\n  📋 Demo Query {i}: {query_config['description']}")
            print(f"     Question: {query_config['question']}")
            
            try:
                result = rag_engine.query(
                    question=query_config["question"],
                    client_id=query_config.get("client_id"),
                    enable_cross_client=query_config.get("enable_cross_client", False),
                    k=3
                )
                
                answer = result["answer"]
                if len(answer) > 200:
                    answer = answer[:200] + "..."
                
                print(f"     Answer: {answer}")
                print(f"     Sources: {len(result['sources'])} documents")
                
            except Exception as e:
                print(f"     ❌ Error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Demo queries failed: {str(e)}")
        return False


def generate_sample_report():
    """Generate a sample report."""
    print("\n📊 Generating sample report...")
    
    try:
        from financial_kb.src.generation.report_generator import FinancialReportGenerator, ReportConfig
        
        generator = FinancialReportGenerator()
        
        config = ReportConfig(
            client_id="AAPL",
            report_type="executive",
            output_format="markdown"
        )
        
        result = generator.generate_client_report("AAPL", config)
        
        if "error" not in result:
            output_file = result.get("output_file", "Unknown")
            print(f"  ✅ Sample report generated: {output_file}")
            
            sections = list(result.get("sections", {}).keys())
            print(f"  📄 Report sections: {', '.join(sections)}")
            
            return True
        else:
            print(f"  ❌ Report generation failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"  ❌ Sample report failed: {str(e)}")
        return False


def show_next_steps():
    """Show next steps for users."""
    print("\n🎯 Next Steps:")
    print("""
    1. 🌐 Launch Web Interface:
       python -m financial_kb.cli webapp
       
    2. 💾 Download Real SEC Data:
       python -m financial_kb.cli download -t "AAPL,MSFT,GOOGL" -f "10-K,10-Q" -n 2
       
    3. 🔍 Query the System:
       python -m financial_kb.cli query -q "What was Apple's revenue growth?" -c AAPL
       
    4. 📊 Generate Reports:
       python -m financial_kb.cli report -c AAPL --type comprehensive
       
    5. 📈 Comparative Analysis:
       python -m financial_kb.cli report --clients "AAPL,MSFT,GOOGL" --type comparative
    
    📚 Documentation:
       - README.md for detailed setup instructions
       - financial_kb/config/settings.py for configuration options
       - env.example for environment variable template
    
    🆘 Need Help?
       python -m financial_kb.cli status  # Check system status
       python -m financial_kb.cli --help  # View all CLI options
    """)


def main():
    """Main quick start function."""
    print_banner()
    
    # Check prerequisites
    issues, recommendations = check_prerequisites()
    
    if issues:
        print("\n❌ Prerequisites check failed:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues before continuing.")
        sys.exit(1)
    
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   - {rec}")
    
    # Initialize system
    if not initialize_system():
        print("\nSystem initialization failed. Please check the logs and try again.")
        sys.exit(1)
    
    # Run demonstrations
    print("\n🎬 Running demonstrations...")
    
    # Demo queries
    query_success = run_demo_queries()
    
    # Sample report
    report_success = generate_sample_report()
    
    # Summary
    print("\n" + "="*60)
    print("📋 Quick Start Summary")
    print("="*60)
    
    if query_success and report_success:
        print("✅ All demonstrations completed successfully!")
        print("🎉 Your Financial Knowledge Base RAG System is ready to use!")
    else:
        print("⚠️  Some demonstrations encountered issues.")
        print("   The system is still functional, but you may want to check the configuration.")
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()