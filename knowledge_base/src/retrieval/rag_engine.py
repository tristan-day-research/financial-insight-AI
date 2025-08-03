"""
RAG (Retrieval-Augmented Generation) engine for financial document querying.
Combines vector search with LLM generation for accurate financial insights.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import json
from datetime import datetime
import re

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever

from knowledge_base.config.settings import get_settings, validate_api_keys
from knowledge_base.src.storage.vector_store import FinancialVectorStore
from knowledge_base.src.storage.sql_store import FinancialSQLStore

logger = logging.getLogger(__name__)


class FinancialRAGRetriever(BaseRetriever):
    """Custom retriever that combines vector search with SQL metadata filtering."""
    
    def __init__(
        self,
        vector_store: FinancialVectorStore,
        sql_store: FinancialSQLStore,
        default_k: int = 5
    ):
        super().__init__()
        self.vector_store = vector_store
        self.sql_store = sql_store
        self.default_k = default_k
    
    def get_relevant_documents(
        self,
        query: str,
        client_id: Optional[str] = None,
        enable_cross_client: bool = False,
        document_filters: Optional[Dict] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve relevant documents with enhanced filtering."""
        k = k or self.default_k
        
        # Build metadata filters
        metadata_filters = {}
        if document_filters:
            metadata_filters.update(document_filters)
        
        # Perform vector search
        documents = self.vector_store.similarity_search(
            query=query,
            k=k * 2,  # Get extra results for SQL filtering
            client_id=client_id,
            enable_cross_client=enable_cross_client,
            filter_metadata=metadata_filters
        )
        
        # Enhanced filtering using SQL metadata
        if document_filters or client_id:
            filtered_docs = self._apply_sql_filters(documents, document_filters, client_id)
            return filtered_docs[:k]
        
        return documents[:k]
    
    def _apply_sql_filters(
        self,
        documents: List[Document],
        filters: Optional[Dict],
        client_id: Optional[str]
    ) -> List[Document]:
        """Apply additional SQL-based filtering to retrieved documents."""
        if not documents:
            return documents
        
        # Extract document IDs from retrieved documents
        doc_ids = []
        for doc in documents:
            doc_id = doc.metadata.get('document_id')
            if doc_id:
                doc_ids.append(doc_id)
        
        if not doc_ids:
            return documents
        
        # Query SQL database for additional metadata
        session = self.sql_store.get_session()
        try:
            from knowledge_base.src.storage.sql_store import Document as SQLDocument
            
            query = session.query(SQLDocument).filter(
                SQLDocument.document_id.in_(doc_ids)
            )
            
            if client_id:
                query = query.filter(SQLDocument.client_id == client_id)
            
            # Apply additional filters
            if filters:
                if 'filing_types' in filters:
                    query = query.filter(SQLDocument.filing_type.in_(filters['filing_types']))
                
                if 'has_financial_data' in filters and filters['has_financial_data']:
                    query = query.filter(
                        (SQLDocument.has_revenue_data == True) |
                        (SQLDocument.has_profit_data == True) |
                        (SQLDocument.has_balance_sheet == True) |
                        (SQLDocument.has_cash_flow == True)
                    )
                
                if 'min_financial_density' in filters:
                    query = query.filter(
                        SQLDocument.financial_density >= filters['min_financial_density']
                    )
            
            sql_docs = query.all()
            valid_doc_ids = {doc.document_id for doc in sql_docs}
            
            # Filter original documents based on SQL results
            filtered_documents = [
                doc for doc in documents
                if doc.metadata.get('document_id') in valid_doc_ids
            ]
            
            return filtered_documents
            
        except Exception as e:
            logger.warning(f"Error applying SQL filters: {str(e)}")
            return documents
        finally:
            session.close()


class FinancialRAGEngine:
    """
    Main RAG engine for financial knowledge base queries.
    Supports client isolation, federated queries, and financial domain expertise.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize storage components
        self.vector_store = FinancialVectorStore()
        self.sql_store = FinancialSQLStore()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize custom retriever
        self.retriever = FinancialRAGRetriever(
            vector_store=self.vector_store,
            sql_store=self.sql_store,
            default_k=5
        )
        
        # Initialize memory for conversational queries
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Financial domain prompts
        self.prompts = self._initialize_prompts()
        
        # Initialize QA chains
        self.qa_chain = self._initialize_qa_chain()
        self.conversational_chain = self._initialize_conversational_chain()
    
    def _initialize_llm(self):
        """Initialize the language model with fallback support."""
        api_validation = validate_api_keys()
        
        if api_validation["openai_available"]:
            try:
                llm = ChatOpenAI(
                    model_name=self.settings.api.openai_model,
                    temperature=0.1,  # Low temperature for factual financial queries
                    openai_api_key=self.settings.api.openai_api_key
                )
                logger.info(f"Using OpenAI model: {self.settings.api.openai_model}")
                return llm
            except Exception as e:
                logger.warning(f"OpenAI LLM failed: {e}. Falling back to local model.")
        
        # Fallback to local model (requires significant resources)
        logger.warning("OpenAI not available. Limited functionality with local models.")
        # For MVP, we'll return a placeholder that raises an error for generation
        return None
    
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize financial domain-specific prompts."""
        
        # Standard QA prompt for financial queries
        qa_template = """You are a financial analyst AI assistant helping with analysis of SEC filings and financial documents.

Use the following pieces of context to answer the question. The context comes from verified financial documents.

Key Guidelines:
1. Be precise and factual - only use information from the provided context
2. For financial figures, include units (millions, billions, etc.) and time periods
3. If asked about trends, compare across time periods when data is available
4. Acknowledge limitations if context doesn't contain sufficient information
5. For client-specific queries, focus only on that company's data
6. For comparative queries, clearly distinguish between different companies

Context:
{context}

Question: {question}

Provide a comprehensive, well-structured answer:"""

        # Comparative analysis prompt for cross-client queries  
        comparative_template = """You are a financial analyst conducting comparative analysis across multiple companies.

Use the provided context from various company filings to perform the comparative analysis.

Key Guidelines:
1. Clearly identify which company each data point belongs to
2. Present comparisons in a structured format (tables, bullet points)
3. Highlight key differences and similarities
4. Include relevant financial ratios and metrics when possible
5. Note any limitations in comparability (different fiscal years, accounting methods, etc.)
6. Provide insights on relative performance

Context from multiple companies:
{context}

Comparative Analysis Request: {question}

Provide a structured comparative analysis:"""

        # Executive summary prompt for high-level insights
        executive_template = """You are preparing an executive summary based on financial document analysis.

Create a concise, executive-level summary that highlights key insights and actionable information.

Key Guidelines:
1. Start with the most important findings
2. Use business language appropriate for executives
3. Include specific financial figures that support conclusions
4. Highlight risks and opportunities
5. Keep technical jargon to a minimum
6. Focus on strategic implications

Context:
{context}

Executive Summary Request: {question}

Provide an executive summary:"""

        return {
            "standard_qa": PromptTemplate(
                template=qa_template,
                input_variables=["context", "question"]
            ),
            "comparative": PromptTemplate(
                template=comparative_template,
                input_variables=["context", "question"]
            ),
            "executive": PromptTemplate(
                template=executive_template,
                input_variables=["context", "question"]
            )
        }
    
    def _initialize_qa_chain(self):
        """Initialize the QA chain for standard queries."""
        if not self.llm:
            return None
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self.prompts["standard_qa"]
            },
            return_source_documents=True
        )
    
    def _initialize_conversational_chain(self):
        """Initialize conversational chain for follow-up queries."""
        if not self.llm:
            return None
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True
        )
    
    def query(
        self,
        question: str,
        client_id: Optional[str] = None,
        query_type: str = "standard",
        enable_cross_client: bool = False,
        filters: Optional[Dict] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a query with client isolation and response generation.
        
        Args:
            question: The question to ask
            client_id: Optional client ID for filtering
            query_type: Type of query (standard, comparative, executive)
            enable_cross_client: Allow cross-client search
            filters: Additional metadata filters
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.llm:
            return self._fallback_response(question, client_id, enable_cross_client, filters, k)
        
        try:
            # Validate query type
            if query_type not in ["standard", "comparative", "executive"]:
                query_type = "standard"
            
            # Retrieve relevant documents
            documents = self.retriever.get_relevant_documents(
                query=question,
                client_id=client_id,
                enable_cross_client=enable_cross_client,
                document_filters=filters,
                k=k
            )
            
            if not documents:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or check if the relevant documents have been ingested.",
                    "sources": [],
                    "query_metadata": {
                        "client_id": client_id,
                        "query_type": query_type,
                        "cross_client": enable_cross_client,
                        "documents_found": 0
                    }
                }
            
            # Select appropriate prompt based on query type
            if query_type == "comparative" and enable_cross_client:
                prompt = self.prompts["comparative"]
            elif query_type == "executive":
                prompt = self.prompts["executive"]
            else:
                prompt = self.prompts["standard_qa"]
            
            # Generate response using the appropriate chain
            if query_type == "standard":
                result = self.qa_chain({
                    "query": question,
                    "retriever_kwargs": {
                        "client_id": client_id,
                        "enable_cross_client": enable_cross_client,
                        "document_filters": filters,
                        "k": k
                    }
                })
            else:
                # For specialized queries, manually format the context
                context = self._format_context(documents, query_type)
                formatted_prompt = prompt.format(context=context, question=question)
                
                response = self.llm.predict(formatted_prompt)
                result = {
                    "answer": response,
                    "source_documents": documents
                }
            
            # Format response with enhanced metadata
            return self._format_response(result, question, client_id, query_type, enable_cross_client)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "query_metadata": {
                    "error": str(e),
                    "client_id": client_id,
                    "query_type": query_type
                }
            }
    
    def _fallback_response(
        self,
        question: str,
        client_id: Optional[str],
        enable_cross_client: bool,
        filters: Optional[Dict],
        k: int
    ) -> Dict[str, Any]:
        """Provide fallback response when LLM is not available."""
        # Retrieve documents without generation
        documents = self.retriever.get_relevant_documents(
            query=question,
            client_id=client_id,
            enable_cross_client=enable_cross_client,
            document_filters=filters,
            k=k
        )
        
        # Create a simple response based on retrieved content
        if documents:
            content_summary = "\n\n".join([
                f"**Document {i+1}** (from {doc.metadata.get('client_id', 'Unknown')}):\n{doc.page_content[:300]}..."
                for i, doc in enumerate(documents[:3])
            ])
            
            answer = f"I found {len(documents)} relevant documents. Here are the key excerpts:\n\n{content_summary}\n\n*Note: Full AI analysis unavailable - please configure OpenAI API key for complete responses.*"
        else:
            answer = "No relevant documents found for your query."
        
        return {
            "answer": answer,
            "sources": [self._format_source(doc) for doc in documents],
            "query_metadata": {
                "client_id": client_id,
                "query_type": "fallback",
                "cross_client": enable_cross_client,
                "documents_found": len(documents),
                "llm_available": False
            }
        }
    
    def _format_context(self, documents: List[Document], query_type: str) -> str:
        """Format retrieved documents into context for the prompt."""
        if query_type == "comparative":
            # Group by client for comparative analysis
            client_groups = {}
            for doc in documents:
                client_id = doc.metadata.get('client_id', 'Unknown')
                if client_id not in client_groups:
                    client_groups[client_id] = []
                client_groups[client_id].append(doc)
            
            context_parts = []
            for client_id, client_docs in client_groups.items():
                client_context = f"\n--- {client_id} ---\n"
                client_context += "\n".join([doc.page_content for doc in client_docs])
                context_parts.append(client_context)
            
            return "\n\n".join(context_parts)
        else:
            # Standard context formatting
            return "\n\n".join([
                f"Document {i+1} (Source: {doc.metadata.get('client_id', 'Unknown')} - {doc.metadata.get('filing_type', 'Unknown')}):\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ])
    
    def _format_response(
        self,
        result: Dict,
        question: str,
        client_id: Optional[str],
        query_type: str,
        enable_cross_client: bool
    ) -> Dict[str, Any]:
        """Format the final response with metadata."""
        sources = []
        if "source_documents" in result:
            sources = [self._format_source(doc) for doc in result["source_documents"]]
        
        return {
            "answer": result.get("answer", "No answer generated"),
            "sources": sources,
            "query_metadata": {
                "question": question,
                "client_id": client_id,
                "query_type": query_type,
                "cross_client": enable_cross_client,
                "documents_found": len(sources),
                "timestamp": datetime.now().isoformat(),
                "llm_available": True
            }
        }
    
    def _format_source(self, document: Document) -> Dict[str, Any]:
        """Format source document information."""
        return {
            "content": document.page_content[:200] + "..." if len(document.page_content) > 200 else document.page_content,
            "metadata": {
                "client_id": document.metadata.get("client_id"),
                "filing_type": document.metadata.get("filing_type"),
                "section": document.metadata.get("section"),
                "document_id": document.metadata.get("document_id"),
                "filing_date": document.metadata.get("filing_date"),
                "content_type": document.metadata.get("content_type"),
                "financial_density": document.metadata.get("financial_density")
            }
        }
    
    def get_client_summary(self, client_id: str) -> Dict[str, Any]:
        """Generate a comprehensive summary for a specific client."""
        summary_query = f"Provide a comprehensive financial overview and analysis for {client_id}, including key metrics, financial position, and recent developments."
        
        return self.query(
            question=summary_query,
            client_id=client_id,
            query_type="executive",
            k=10
        )
    
    def compare_clients(self, client_ids: List[str], comparison_aspect: str = "financial performance") -> Dict[str, Any]:
        """Perform comparative analysis across multiple clients."""
        clients_str = ", ".join(client_ids)
        comparison_query = f"Compare the {comparison_aspect} of {clients_str}. Provide a detailed analysis highlighting similarities, differences, and relative strengths."
        
        return self.query(
            question=comparison_query,
            query_type="comparative",
            enable_cross_client=True,
            filters={"client_ids": client_ids},
            k=15
        )
    
    def get_financial_trends(self, client_id: Optional[str] = None, metric: str = "revenue") -> Dict[str, Any]:
        """Analyze financial trends for specific metrics."""
        if client_id:
            trend_query = f"Analyze the {metric} trends for {client_id} over time. Include year-over-year changes, growth rates, and any notable patterns."
        else:
            trend_query = f"Analyze {metric} trends across all companies in the database. Identify industry patterns and outliers."
        
        return self.query(
            question=trend_query,
            client_id=client_id,
            enable_cross_client=(client_id is None),
            query_type="standard",
            k=8
        )


def main():
    """Example usage of RAG engine."""
    rag_engine = FinancialRAGEngine()
    
    # Example queries
    queries = [
        {
            "question": "What was Apple's revenue in the latest filing?",
            "client_id": "AAPL",
            "query_type": "standard"
        },
        {
            "question": "Compare the profitability of Apple, Microsoft, and Google",
            "query_type": "comparative",
            "enable_cross_client": True
        },
        {
            "question": "Provide an executive summary of Microsoft's recent financial performance",
            "client_id": "MSFT",
            "query_type": "executive"
        }
    ]
    
    for i, query_config in enumerate(queries):
        print(f"\n--- Query {i+1} ---")
        print(f"Question: {query_config['question']}")
        
        result = rag_engine.query(**query_config)
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources found: {len(result['sources'])}")
        print(f"Query metadata: {result['query_metadata']}")


if __name__ == "__main__":
    main()