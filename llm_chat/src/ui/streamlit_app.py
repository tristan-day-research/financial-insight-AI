"""
Streamlit web interface for Financial Knowledge Base RAG System.
Provides interactive demo of document ingestion, querying, and report generation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config.settings import get_settings, validate_api_keys
from knowledge_base.src.ingestion.sec_downloader import SECDownloader
from knowledge_base.src.ingestion.document_processor import FinancialDocumentProcessor
from knowledge_base.src.storage.vector_store import FinancialVectorStore
from knowledge_base.src.storage.sql_store import FinancialSQLStore
from knowledge_base.src.retrieval.rag_engine import FinancialRAGEngine
from knowledge_base.src.generation.report_generator import FinancialReportGenerator, ReportConfig


# Page configuration
st.set_page_config(
    page_title="Financial Knowledge Base RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_components():
    """Initialize system components with caching."""
    try:
        settings = get_settings()
        api_status = validate_api_keys()
        
        # Initialize components
        downloader = SECDownloader()
        processor = FinancialDocumentProcessor()
        vector_store = FinancialVectorStore()
        sql_store = FinancialSQLStore()
        rag_engine = FinancialRAGEngine()
        report_generator = FinancialReportGenerator()
        
        return {
            "downloader": downloader,
            "processor": processor,
            "vector_store": vector_store,
            "sql_store": sql_store,
            "rag_engine": rag_engine,
            "report_generator": report_generator,
            "api_status": api_status,
            "settings": settings
        }
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None


def show_header():
    """Display application header and status."""
    st.title("📊 Financial Knowledge Base RAG System")
    st.markdown("### Enterprise AI for Financial Document Analysis")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.get("api_status", {}).get("openai_available", False):
            st.success("✅ OpenAI API Connected")
        else:
            st.warning("⚠️ OpenAI API Not Available (Fallback Mode)")
    
    with col2:
        if st.session_state.get("components"):
            st.success("✅ System Initialized")
        else:
            st.error("❌ System Error")
    
    with col3:
        clients = st.session_state.get("available_clients", [])
        st.info(f"📁 {len(clients)} Clients Available")


def show_sidebar():
    """Display sidebar with navigation and controls."""
    st.sidebar.title("Navigation")
    
    pages = {
        "🏠 Dashboard": "dashboard",
        "📥 Data Ingestion": "ingestion",
        "🔍 Query Engine": "query",
        "📊 Report Generation": "reports",
        "⚙️ Settings": "settings"
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Page",
        list(pages.keys()),
        index=0
    )
    
    st.session_state["current_page"] = pages[selected_page]
    
    # Client selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Client Selection")
    
    clients = st.session_state.get("available_clients", [])
    if clients:
        selected_client = st.sidebar.selectbox(
            "Select Client",
            ["All Clients"] + clients,
            index=0
        )
        st.session_state["selected_client"] = None if selected_client == "All Clients" else selected_client
    else:
        st.sidebar.info("No clients available. Please ingest some data first.")
        st.session_state["selected_client"] = None
    
    # Quick stats
    if clients:
        st.sidebar.markdown("### Quick Stats")
        components = st.session_state.get("components")
        if components:
            try:
                stats_df = components["sql_store"].get_client_statistics()
                if not stats_df.empty:
                    total_docs = stats_df["document_count"].sum()
                    total_chunks = stats_df["chunk_count"].sum()
                    st.sidebar.metric("Total Documents", int(total_docs))
                    st.sidebar.metric("Total Chunks", int(total_chunks))
            except Exception as e:
                st.sidebar.error(f"Error loading stats: {str(e)}")


def show_dashboard():
    """Display main dashboard with system overview."""
    st.header("System Dashboard")
    
    components = st.session_state.get("components")
    if not components:
        st.error("System components not initialized")
        return
    
    try:
        # Get client statistics
        stats_df = components["sql_store"].get_client_statistics()
        
        if stats_df.empty:
            st.info("No data available. Please ingest some documents first.")
            return
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_clients = len(stats_df)
            st.metric("Total Clients", total_clients)
        
        with col2:
            total_docs = int(stats_df["document_count"].sum())
            st.metric("Total Documents", total_docs)
        
        with col3:
            total_chunks = int(stats_df["chunk_count"].sum())
            st.metric("Total Chunks", total_chunks)
        
        with col4:
            avg_density = stats_df["avg_financial_density"].mean()
            st.metric("Avg Financial Density", f"{avg_density:.1f}%")
        
        # Client overview table
        st.subheader("Client Overview")
        display_df = stats_df[["client_id", "company_name", "industry", "document_count", "chunk_count"]].copy()
        st.dataframe(display_df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Documents by Client")
            fig = px.bar(
                stats_df,
                x="client_id",
                y="document_count",
                title="Number of Documents per Client"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Industry Distribution")
            industry_counts = stats_df["industry"].value_counts()
            fig = px.pie(
                values=industry_counts.values,
                names=industry_counts.index,
                title="Clients by Industry"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")


def show_ingestion():
    """Display data ingestion interface."""
    st.header("Data Ingestion")
    
    components = st.session_state.get("components")
    if not components:
        st.error("System components not initialized")
        return
    
    tab1, tab2, tab3 = st.tabs(["SEC Filings", "Document Processing", "Ingestion Status"])
    
    with tab1:
        st.subheader("Download SEC Filings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tickers = st.text_input(
                "Company Tickers (comma-separated)",
                value="AAPL,MSFT,GOOGL",
                help="Enter stock ticker symbols separated by commas"
            )
            
            filing_types = st.multiselect(
                "Filing Types",
                ["10-K", "10-Q", "8-K"],
                default=["10-K", "10-Q"]
            )
        
        with col2:
            num_filings = st.slider(
                "Number of filings per type",
                min_value=1,
                max_value=10,
                value=2
            )
            
            start_download = st.button("Start Download", type="primary")
        
        if start_download and tickers:
            ticker_list = [t.strip().upper() for t in tickers.split(",")]
            
            with st.spinner("Downloading SEC filings..."):
                try:
                    results = components["downloader"].bulk_download_companies(
                        tickers=ticker_list,
                        filing_types=filing_types,
                        num_filings=num_filings
                    )
                    
                    st.success(f"Downloaded filings for {len(results)} companies")
                    
                    # Display results
                    for ticker, filings in results.items():
                        st.write(f"**{ticker}**: {len(filings)} filings downloaded")
                        if filings:
                            df = pd.DataFrame(filings)
                            st.dataframe(df[["filing_type", "filing_date", "file_size"]], use_container_width=True)
                    
                    # Store results for processing
                    st.session_state["download_results"] = results
                    
                except Exception as e:
                    st.error(f"Error downloading filings: {str(e)}")
    
    with tab2:
        st.subheader("Process Documents")
        
        download_results = st.session_state.get("download_results")
        
        if download_results:
            st.info(f"Ready to process documents for {len(download_results)} companies")
            
            process_docs = st.button("Process and Index Documents", type="primary")
            
            if process_docs:
                with st.spinner("Processing documents and creating embeddings..."):
                    try:
                        total_chunks = 0
                        
                        for ticker, filings in download_results.items():
                            if not filings:
                                continue
                            
                            st.write(f"Processing {ticker}...")
                            
                            # Process documents
                            chunks = components["processor"].process_multiple_documents(filings)
                            
                            if chunks:
                                # Add to vector store
                                doc_ids = components["vector_store"].add_documents(chunks, client_id=ticker)
                                
                                # Add client to SQL store
                                client_data = {
                                    "id": ticker,
                                    "company_name": f"{ticker} Corporation",
                                    "industry": "Technology",  # Would be enhanced with real data
                                    "sector": "Technology"
                                }
                                components["sql_store"].add_client(client_data)
                                
                                # Add documents to SQL store
                                for filing in filings:
                                    components["sql_store"].add_document(filing)
                                
                                total_chunks += len(chunks)
                                st.write(f"  ✅ {len(chunks)} chunks indexed")
                        
                        st.success(f"Processing complete! {total_chunks} total chunks indexed.")
                        
                        # Refresh available clients
                        refresh_available_clients()
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
        else:
            st.info("Please download SEC filings first in the previous tab.")
    
    with tab3:
        st.subheader("Ingestion Status")
        
        try:
            stats_df = components["sql_store"].get_client_statistics()
            
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No ingested data found.")
        except Exception as e:
            st.error(f"Error loading ingestion status: {str(e)}")


def show_query():
    """Display query interface."""
    st.header("RAG Query Engine")
    
    components = st.session_state.get("components")
    if not components:
        st.error("System components not initialized")
        return
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Ask a question about the financial documents",
            value="What was the revenue growth for the selected company?",
            height=100
        )
    
    with col2:
        query_type = st.selectbox(
            "Query Type",
            ["standard", "executive", "comparative"]
        )
        
        enable_cross_client = st.checkbox(
            "Enable Cross-Client Search",
            value=False,
            help="Allow searching across all companies for comparative analysis"
        )
        
        num_results = st.slider("Number of results", 1, 10, 5)
    
    # Execute query
    if st.button("Execute Query", type="primary") and query:
        selected_client = st.session_state.get("selected_client")
        
        with st.spinner("Processing query..."):
            try:
                result = components["rag_engine"].query(
                    question=query,
                    client_id=selected_client,
                    query_type=query_type,
                    enable_cross_client=enable_cross_client,
                    k=num_results
                )
                
                # Display results
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Query metadata
                with st.expander("Query Metadata"):
                    st.json(result["query_metadata"])
                
                # Sources
                if result["sources"]:
                    st.subheader("Sources")
                    for i, source in enumerate(result["sources"]):
                        with st.expander(f"Source {i+1} - {source['metadata'].get('client_id', 'Unknown')}"):
                            st.write("**Content:**")
                            st.write(source["content"])
                            st.write("**Metadata:**")
                            st.json(source["metadata"])
                
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
    
    # Predefined queries
    st.subheader("Sample Queries")
    
    sample_queries = [
        "What was the revenue for the latest fiscal year?",
        "What are the main risk factors mentioned?",
        "How did profitability change year over year?",
        "What are the key business segments?",
        "Compare revenue growth across all companies",
        "What are the main competitive advantages mentioned?"
    ]
    
    for i, sample_query in enumerate(sample_queries):
        if st.button(sample_query, key=f"sample_{i}"):
            st.session_state["sample_query"] = sample_query
            st.experimental_rerun()
    
    # Auto-fill sample query
    if "sample_query" in st.session_state:
        st.text_area(
            "Selected sample query",
            value=st.session_state["sample_query"],
            key="query_area"
        )
        del st.session_state["sample_query"]


def show_reports():
    """Display report generation interface."""
    st.header("Report Generation")
    
    components = st.session_state.get("components")
    if not components:
        st.error("System components not initialized")
        return
    
    tab1, tab2 = st.tabs(["Generate Reports", "View Reports"])
    
    with tab1:
        report_type = st.selectbox(
            "Report Type",
            ["comprehensive", "executive", "comparative", "trend"]
        )
        
        if report_type == "comparative":
            st.subheader("Comparative Report")
            available_clients = st.session_state.get("available_clients", [])
            
            if len(available_clients) < 2:
                st.warning("Need at least 2 clients for comparative analysis")
                return
            
            selected_clients = st.multiselect(
                "Select Companies to Compare",
                available_clients,
                default=available_clients[:3] if len(available_clients) >= 3 else available_clients
            )
            
            if st.button("Generate Comparative Report", type="primary") and len(selected_clients) >= 2:
                with st.spinner("Generating comparative report..."):
                    try:
                        config = ReportConfig(
                            report_type="comparative",
                            output_format="markdown"
                        )
                        
                        report = components["report_generator"].generate_comparative_report(
                            client_ids=selected_clients,
                            config=config
                        )
                        
                        if "error" not in report:
                            st.success("Comparative report generated successfully!")
                            
                            # Display report sections
                            for section_name, section_data in report["sections"].items():
                                st.subheader(section_name.replace("_", " ").title())
                                st.write(section_data["content"])
                            
                            # Download link
                            if report.get("output_file"):
                                st.download_button(
                                    label="Download Report",
                                    data=open(report["output_file"], "r").read(),
                                    file_name=Path(report["output_file"]).name,
                                    mime="text/markdown"
                                )
                        else:
                            st.error(f"Error generating report: {report['error']}")
                    
                    except Exception as e:
                        st.error(f"Error generating comparative report: {str(e)}")
        
        else:
            st.subheader("Client-Specific Report")
            selected_client = st.session_state.get("selected_client")
            
            if not selected_client:
                st.warning("Please select a client in the sidebar")
                return
            
            if st.button("Generate Client Report", type="primary"):
                with st.spinner(f"Generating {report_type} report for {selected_client}..."):
                    try:
                        config = ReportConfig(
                            client_id=selected_client,
                            report_type=report_type,
                            output_format="markdown"
                        )
                        
                        report = components["report_generator"].generate_client_report(
                            client_id=selected_client,
                            config=config
                        )
                        
                        if "error" not in report:
                            st.success("Report generated successfully!")
                            
                            # Display report sections
                            for section_name, section_data in report["sections"].items():
                                st.subheader(section_name.replace("_", " ").title())
                                st.write(section_data["content"])
                            
                            # Download link
                            if report.get("output_file"):
                                st.download_button(
                                    label="Download Report",
                                    data=open(report["output_file"], "r").read(),
                                    file_name=Path(report["output_file"]).name,
                                    mime="text/markdown"
                                )
                        else:
                            st.error(f"Error generating report: {report['error']}")
                    
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
    
    with tab2:
        st.subheader("Generated Reports")
        
        # List generated reports
        output_dir = Path(components["report_generator"].output_dir)
        if output_dir.exists():
            report_files = list(output_dir.glob("*.md"))
            
            if report_files:
                for report_file in sorted(report_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"📄 {report_file.name}")
                        st.caption(f"Modified: {datetime.fromtimestamp(report_file.stat().st_mtime)}")
                    
                    with col2:
                        if st.button("View", key=f"view_{report_file.name}"):
                            with open(report_file, "r") as f:
                                st.markdown(f.read())
                    
                    with col3:
                        st.download_button(
                            label="Download",
                            data=open(report_file, "r").read(),
                            file_name=report_file.name,
                            mime="text/markdown",
                            key=f"download_{report_file.name}"
                        )
            else:
                st.info("No reports generated yet.")
        else:
            st.info("No reports directory found.")


def show_settings():
    """Display settings and configuration."""
    st.header("Settings & Configuration")
    
    components = st.session_state.get("components")
    settings = components["settings"] if components else None
    
    if not settings:
        st.error("Settings not available")
        return
    
    tab1, tab2, tab3 = st.tabs(["API Configuration", "System Status", "Data Management"])
    
    with tab1:
        st.subheader("API Configuration")
        
        api_status = st.session_state.get("api_status", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**OpenAI API**")
            if api_status.get("openai_available"):
                st.success("✅ Connected")
                st.write(f"Model: {settings.api.openai_model}")
                st.write(f"Embedding Model: {settings.api.embedding_model}")
            else:
                st.error("❌ Not connected")
                st.write("Please set OPENAI_API_KEY environment variable")
        
        with col2:
            st.write("**Embedding Configuration**")
            st.write(f"Vector DB: {settings.database.vector_db_type}")
            st.write(f"Fallback Model: {settings.api.fallback_embedding_model}")
    
    with tab2:
        st.subheader("System Status")
        
        try:
            # Vector store statistics
            if components:
                clients = components["vector_store"].list_clients()
                st.write(f"**Vector Store**: {len(clients)} clients indexed")
                
                # SQL store statistics
                stats_df = components["sql_store"].get_client_statistics()
                if not stats_df.empty:
                    total_docs = int(stats_df["document_count"].sum())
                    total_chunks = int(stats_df["chunk_count"].sum())
                    st.write(f"**SQL Store**: {total_docs} documents, {total_chunks} chunks")
        
        except Exception as e:
            st.error(f"Error loading system status: {str(e)}")
    
    with tab3:
        st.subheader("Data Management")
        
        if components:
            available_clients = st.session_state.get("available_clients", [])
            
            if available_clients:
                st.write("**Available Clients:**")
                for client in available_clients:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• {client}")
                    with col2:
                        if st.button(f"Delete {client}", key=f"delete_{client}"):
                            if st.checkbox(f"Confirm deletion of {client}", key=f"confirm_{client}"):
                                try:
                                    components["vector_store"].delete_client_data(client)
                                    components["sql_store"].delete_client_data(client)
                                    st.success(f"Deleted data for {client}")
                                    refresh_available_clients()
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Error deleting {client}: {str(e)}")
            else:
                st.info("No client data available")


def refresh_available_clients():
    """Refresh the list of available clients."""
    components = st.session_state.get("components")
    if components:
        try:
            clients = components["vector_store"].list_clients()
            st.session_state["available_clients"] = clients
        except Exception as e:
            st.error(f"Error refreshing clients: {str(e)}")
            st.session_state["available_clients"] = []


def main():
    """Main application entry point."""
    # Initialize session state
    if "components" not in st.session_state:
        with st.spinner("Initializing Financial Knowledge Base system..."):
            st.session_state["components"] = initialize_components()
            if st.session_state["components"]:
                st.session_state["api_status"] = st.session_state["components"]["api_status"]
                refresh_available_clients()
    
    # Show header
    show_header()
    
    # Show sidebar
    show_sidebar()
    
    # Route to selected page
    current_page = st.session_state.get("current_page", "dashboard")
    
    if current_page == "dashboard":
        show_dashboard()
    elif current_page == "ingestion":
        show_ingestion()
    elif current_page == "query":
        show_query()
    elif current_page == "reports":
        show_reports()
    elif current_page == "settings":
        show_settings()
    
    # Footer
    st.markdown("---")
    st.markdown("*Financial Knowledge Base RAG System - Built with Streamlit*")


if __name__ == "__main__":
    main()