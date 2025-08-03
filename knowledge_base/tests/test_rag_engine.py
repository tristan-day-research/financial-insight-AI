"""
Unit tests for RAG engine functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from knowledge_base.src.retrieval.rag_engine import FinancialRAGEngine


class TestRAGEngine:
    """Test cases for RAGEngine class."""
    
    @pytest.fixture
    def rag_engine(self, test_settings):
        """Create a test instance of FinancialRAGEngine."""
        return FinancialRAGEngine()
    
    def test_initialization(self, rag_engine):
        """Test FinancialRAGEngine initialization."""
        assert rag_engine is not None
        assert hasattr(rag_engine, 'vector_store')
        assert hasattr(rag_engine, 'sql_store')
        assert hasattr(rag_engine, 'llm')
    
    @patch('knowledge_base.src.retrieval.rag_engine.FinancialVectorStore')
    @patch('knowledge_base.src.retrieval.rag_engine.FinancialSQLStore')
    @patch('knowledge_base.src.retrieval.rag_engine.ChatOpenAI')
    def test_components_initialization(self, mock_llm, mock_sql, mock_vector, test_settings):
        """Test that all RAG components are properly initialized."""
        # Mock the components
        mock_vector.return_value = Mock()
        mock_sql.return_value = Mock()
        mock_llm.return_value = Mock()
        
        rag_engine = FinancialRAGEngine()
        
        # Verify components were initialized
        mock_vector.assert_called_once()
        mock_sql.assert_called_once()
        mock_llm.assert_called_once()
    
    def test_search_documents(self, rag_engine):
        """Test document search functionality."""
        # Mock the vector store search
        mock_results = [
            Mock(
                page_content="Apple Inc. reported revenue of $394.3 billion.",
                metadata={'ticker': 'AAPL', 'filing_type': '10-K'}
            ),
            Mock(
                page_content="Net income was $96.9 billion for fiscal year 2022.",
                metadata={'ticker': 'AAPL', 'filing_type': '10-K'}
            )
        ]
        rag_engine.vector_store.search.return_value = mock_results
        
        # Test search
        query = "What was Apple's revenue in 2022?"
        results = rag_engine.search_documents(query, top_k=5)
        
        # Verify search was called
        rag_engine.vector_store.search.assert_called_once_with(query, top_k=5)
        
        # Verify results
        assert len(results) == 2
        assert results[0].page_content == "Apple Inc. reported revenue of $394.3 billion."
        assert results[0].metadata['ticker'] == 'AAPL'
    
    def test_get_financial_metrics(self, rag_engine):
        """Test retrieving financial metrics."""
        # Mock SQL store query
        mock_metrics = [
            {
                'metric_name': 'revenue',
                'value': 394328000000,
                'currency': 'USD',
                'period': '2022',
                'ticker': 'AAPL'
            },
            {
                'metric_name': 'net_income',
                'value': 96995000000,
                'currency': 'USD',
                'period': '2022',
                'ticker': 'AAPL'
            }
        ]
        rag_engine.sql_store.get_financial_metrics.return_value = mock_metrics
        
        # Test metric retrieval
        metrics = rag_engine.get_financial_metrics('AAPL', ['revenue', 'net_income'], 2022)
        
        # Verify SQL query was called
        rag_engine.sql_store.get_financial_metrics.assert_called_once_with('AAPL', ['revenue', 'net_income'], 2022)
        
        # Verify results
        assert len(metrics) == 2
        assert metrics[0]['metric_name'] == 'revenue'
        assert metrics[0]['value'] == 394328000000
    
    def test_generate_response(self, rag_engine):
        """Test response generation with context."""
        # Mock LLM response
        mock_llm_response = Mock()
        mock_llm_response.content = "Apple Inc. reported revenue of $394.3 billion in fiscal year 2022."
        rag_engine.llm.invoke.return_value = mock_llm_response
        
        # Mock context retrieval
        mock_context = [
            Mock(
                page_content="Apple Inc. reported revenue of $394.3 billion.",
                metadata={'ticker': 'AAPL', 'filing_type': '10-K'}
            )
        ]
        rag_engine.search_documents = Mock(return_value=mock_context)
        
        # Test response generation
        query = "What was Apple's revenue in 2022?"
        response = rag_engine.generate_response(query)
        
        # Verify LLM was called with context
        rag_engine.llm.invoke.assert_called_once()
        call_args = rag_engine.llm.invoke.call_args[0][0]
        assert query in str(call_args)
        assert "Apple Inc. reported revenue" in str(call_args)
    
    def test_hybrid_search(self, rag_engine):
        """Test hybrid search combining vector and SQL results."""
        # Mock vector search results
        mock_vector_results = [
            Mock(
                page_content="Apple Inc. reported revenue of $394.3 billion.",
                metadata={'ticker': 'AAPL', 'filing_type': '10-K'}
            )
        ]
        rag_engine.search_documents = Mock(return_value=mock_vector_results)
        
        # Mock SQL metrics
        mock_sql_metrics = [
            {
                'metric_name': 'revenue',
                'value': 394328000000,
                'currency': 'USD',
                'period': '2022',
                'ticker': 'AAPL'
            }
        ]
        rag_engine.get_financial_metrics = Mock(return_value=mock_sql_metrics)
        
        # Test hybrid search
        query = "What was Apple's revenue in 2022?"
        results = rag_engine.hybrid_search(query, 'AAPL', ['revenue'], 2022)
        
        # Verify both searches were called
        rag_engine.search_documents.assert_called_once()
        rag_engine.get_financial_metrics.assert_called_once()
        
        # Verify combined results
        assert 'vector_results' in results
        assert 'sql_metrics' in results
        assert len(results['vector_results']) == 1
        assert len(results['sql_metrics']) == 1
    
    def test_analyze_financial_performance(self, rag_engine):
        """Test financial performance analysis."""
        # Mock the analysis components
        rag_engine.get_financial_metrics = Mock(return_value=[
            {'metric_name': 'revenue', 'value': 394328000000, 'period': '2022'},
            {'metric_name': 'net_income', 'value': 96995000000, 'period': '2022'},
            {'metric_name': 'revenue', 'value': 365817000000, 'period': '2021'},
            {'metric_name': 'net_income', 'value': 94680000000, 'period': '2021'}
        ])
        
        rag_engine.generate_response = Mock(return_value="Apple's revenue grew by 7.8% year-over-year.")
        
        # Test analysis
        analysis = rag_engine.analyze_financial_performance('AAPL', 2022)
        
        # Verify analysis was performed
        assert analysis is not None
        assert "revenue grew" in analysis
    
    def test_get_comparative_analysis(self, rag_engine):
        """Test comparative analysis across companies."""
        # Mock comparative data
        mock_comparative_data = {
            'AAPL': [
                {'metric_name': 'revenue', 'value': 394328000000, 'period': '2022'},
                {'metric_name': 'net_income', 'value': 96995000000, 'period': '2022'}
            ],
            'MSFT': [
                {'metric_name': 'revenue', 'value': 198270000000, 'period': '2022'},
                {'metric_name': 'net_income', 'value': 72619000000, 'period': '2022'}
            ]
        }
        
        rag_engine.sql_store.get_comparative_metrics = Mock(return_value=mock_comparative_data)
        rag_engine.generate_response = Mock(return_value="Apple leads in revenue while Microsoft has higher profitability.")
        
        # Test comparative analysis
        analysis = rag_engine.get_comparative_analysis(['AAPL', 'MSFT'], ['revenue', 'net_income'], 2022)
        
        # Verify analysis was performed
        assert analysis is not None
        assert "Apple leads" in analysis or "Microsoft" in analysis
    
    def test_search_with_filters(self, rag_engine):
        """Test search with metadata filters."""
        # Mock filtered search results
        mock_filtered_results = [
            Mock(
                page_content="Apple Inc. 10-K filing for 2022.",
                metadata={'ticker': 'AAPL', 'filing_type': '10-K', 'filing_date': '2022-09-24'}
            )
        ]
        rag_engine.vector_store.search_with_filters = Mock(return_value=mock_filtered_results)
        
        # Test filtered search
        query = "Apple financial performance"
        filters = {'ticker': 'AAPL', 'filing_type': '10-K', 'year': 2022}
        results = rag_engine.search_with_filters(query, filters, top_k=5)
        
        # Verify filtered search was called
        rag_engine.vector_store.search_with_filters.assert_called_once_with(query, filters, top_k=5)
        
        # Verify results
        assert len(results) == 1
        assert results[0].metadata['ticker'] == 'AAPL'
        assert results[0].metadata['filing_type'] == '10-K'


class TestRAGEngineIntegration:
    """Integration tests for RAG engine with real components."""
    
    @pytest.mark.integration
    def test_end_to_end_query_processing(self, test_settings):
        """Test complete end-to-end query processing."""
        rag_engine = RAGEngine(test_settings)
        
        # Test a simple query
        query = "What was Apple's revenue in 2022?"
        
        # This would require actual data in the system
        # For now, we'll test the method exists and doesn't crash
        try:
            response = rag_engine.generate_response(query)
            assert response is not None
        except Exception as e:
            # If no data is available, that's expected
            assert "No data" in str(e) or "No results" in str(e)
    
    @pytest.mark.integration
    def test_financial_metrics_retrieval(self, test_settings):
        """Test retrieving financial metrics from database."""
        rag_engine = RAGEngine(test_settings)
        
        # Test metric retrieval
        try:
            metrics = rag_engine.get_financial_metrics('AAPL', ['revenue'], 2022)
            # If data exists, verify structure
            if metrics:
                assert isinstance(metrics, list)
                for metric in metrics:
                    assert 'metric_name' in metric
                    assert 'value' in metric
                    assert 'ticker' in metric
        except Exception as e:
            # If no data is available, that's expected
            assert "No data" in str(e) or "No results" in str(e)


class TestRAGEnginePerformance:
    """Performance tests for RAG engine."""
    
    def test_search_performance(self, rag_engine):
        """Test search performance with large queries."""
        import time
        
        # Mock vector store for performance testing
        rag_engine.vector_store.search = Mock(return_value=[])
        
        # Test multiple queries
        queries = [
            "What was Apple's revenue in 2022?",
            "What was Microsoft's net income in 2021?",
            "How did Google perform financially in 2023?",
            "What are the key financial metrics for Tesla?",
            "Compare Apple and Microsoft revenue"
        ]
        
        start_time = time.time()
        
        for query in queries:
            rag_engine.search_documents(query, top_k=10)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance is reasonable (should be under 1 second for 5 queries)
        assert total_time < 1.0
        
        # Verify all queries were processed
        assert rag_engine.vector_store.search.call_count == len(queries)
    
    def test_response_generation_performance(self, rag_engine):
        """Test response generation performance."""
        import time
        
        # Mock LLM for performance testing
        mock_response = Mock()
        mock_response.content = "Test response"
        rag_engine.llm.invoke = Mock(return_value=mock_response)
        rag_engine.search_documents = Mock(return_value=[])
        
        # Test response generation
        start_time = time.time()
        
        response = rag_engine.generate_response("Test query")
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Verify response was generated
        assert response is not None
        
        # Verify performance is reasonable (should be under 5 seconds)
        assert generation_time < 5.0 