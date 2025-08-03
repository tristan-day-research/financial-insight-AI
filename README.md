# Financial Knowledge Base RAG System

An enterprise-grade AI system for financial document analysis, built with Retrieval-Augmented Generation (RAG) technology. This system provides comprehensive capabilities for ingesting, processing, and analyzing financial documents with advanced AI-powered insights.

## 🚀 Features

- **Document Ingestion**: Automated SEC filing downloads and document processing
- **Vector Storage**: FAISS, ChromaDB, and Pinecone support with client isolation
- **RAG Engine**: Advanced retrieval and generation with financial context
- **Report Generation**: Comprehensive, executive, and comparative reports
- **Web Interface**: Streamlit-based interactive dashboard
- **CLI Tools**: Command-line interface for automation
- **Multi-format Support**: PDF, DOCX, HTML, and text file processing

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for LLM capabilities)
- Optional: Pinecone API key (for cloud vector storage)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/company/financial-knowledge-base.git
   cd financial-knowledge-base
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

## 🚀 Quick Start

### 1. Initialize the System

```python
from knowledge_base.src.utils.system_init import initialize_demo_system

# Initialize with demo data
initialize_demo_system()
```

### 2. Basic Usage

```python
from knowledge_base.src.ingestion.sec_downloader import SECDownloader
from knowledge_base.src.ingestion.document_processor import FinancialDocumentProcessor
from knowledge_base.src.storage.vector_store import FinancialVectorStore
from knowledge_base.src.storage.sql_store import FinancialSQLStore
from knowledge_base.src.retrieval.rag_engine import FinancialRAGEngine

# Download SEC filings
downloader = SECDownloader()
filings = downloader.download_company_filings("AAPL", ["10-K", "10-Q"], num_filings=2)

# Process documents
processor = FinancialDocumentProcessor()
chunks = processor.process_multiple_documents(filings)

# Store in vector database
vector_store = FinancialVectorStore()
doc_ids = vector_store.add_documents(chunks, client_id="AAPL")

# Query the system
rag_engine = FinancialRAGEngine()
result = rag_engine.query("What was Apple's revenue growth?", client_id="AAPL")
print(result["answer"])
```

### 3. Web Interface

Launch the Streamlit web interface:

```bash
streamlit run knowledge_base/src/ui/streamlit_app.py
```

### 4. CLI Commands

```bash
# Start web interface
python -m knowledge_base.cli webapp

# Download SEC filings
python -m knowledge_base.cli download -t "AAPL,MSFT,GOOGL" -f "10-K,10-Q" -n 2

# Query the system
python -m knowledge_base.cli query -q "What was Apple's revenue growth?" -c AAPL

# Generate reports
python -m knowledge_base.cli report -c AAPL --type comprehensive
python -m knowledge_base.cli report --clients "AAPL,MSFT,GOOGL" --type comparative

# System status
python -m knowledge_base.cli status  # Check system status
python -m knowledge_base.cli --help  # View all CLI options
```

## 📁 Project Structure

```
financial-knowledge-base/
├── knowledge_base/
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── src/
│   │   ├── ingestion/
│   │   │   ├── sec_downloader.py
│   │   │   └── document_processor.py
│   │   ├── storage/
│   │   │   ├── vector_store.py
│   │   │   └── sql_store.py
│   │   ├── retrieval/
│   │   │   └── rag_engine.py
│   │   ├── generation/
│   │   │   └── report_generator.py
│   │   ├── utils/
│   │   │   └── system_init.py
│   │   └── ui/
│   │       └── streamlit_app.py
│   ├── data/
│   │   ├── raw/                 # Raw downloaded files
│   │   ├── processed/           # Processed documents
│   │   └── outputs/             # Generated reports
│   ├── logs/                    # Application logs
│   └── cli.py                   # Command-line interface
├── llm_chat/
│   └── src/
│       └── ui/                  # Moved UI components
├── requirements.txt
├── setup.py
└── README.md
```

## ⚙️ Configuration

The system uses a centralized configuration system. Key settings include:

### Database Configuration
```python
# Vector database settings
DB_VECTOR_DB_TYPE=faiss  # faiss, chromadb, or pinecone
DB_FAISS_INDEX_PATH=knowledge_base/data/faiss_index
DB_CHROMADB_PATH=knowledge_base/data/chromadb
DB_SQLITE_DB_PATH=knowledge_base/data/financial_kb.db

# Pinecone settings (if using Pinecone)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=your_index_name
```

### API Configuration
```python
# OpenAI settings
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002

# SEC API settings
SEC_USER_AGENT=Financial Knowledge Base RAG System
SEC_EMAIL=knowledge_base@company.com
```

### Data Processing
```python
# Document processing
PROC_CHUNK_SIZE=1000
PROC_CHUNK_OVERLAP=200
PROC_MAX_FILE_SIZE_MB=50

# Data paths
DATA_RAW_PATH=knowledge_base/data/raw
DATA_PROCESSED_PATH=knowledge_base/data/processed
DATA_OUTPUT_PATH=knowledge_base/data/outputs
```

See `knowledge_base/config/settings.py` for detailed configuration options including:

- RAG system parameters
- Report generation settings
- Logging configuration
- Security and privacy settings

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v --cov=knowledge_base
```

### Code Formatting
```bash
black knowledge_base/
```

### Linting
```bash
flake8 knowledge_base/
mypy knowledge_base/
```

## 📊 Usage Examples

### 1. Document Ingestion Pipeline

```python
from knowledge_base.src.ingestion.sec_downloader import SECDownloader
from knowledge_base.src.ingestion.document_processor import FinancialDocumentProcessor
from knowledge_base.src.storage.vector_store import FinancialVectorStore

# Download filings
downloader = SECDownloader()
filings = downloader.bulk_download_companies(
    tickers=["AAPL", "MSFT", "GOOGL"],
    filing_types=["10-K", "10-Q"],
    num_filings=3
)

# Process documents
processor = FinancialDocumentProcessor()
for ticker, company_filings in filings.items():
    chunks = processor.process_multiple_documents(company_filings)
    
    # Store in vector database
    vector_store = FinancialVectorStore()
    doc_ids = vector_store.add_documents(chunks, client_id=ticker)
    print(f"Indexed {len(chunks)} chunks for {ticker}")
```

### 2. Advanced Querying

```python
from knowledge_base.src.retrieval.rag_engine import FinancialRAGEngine

rag_engine = FinancialRAGEngine()

# Standard query
result = rag_engine.query(
    question="What was the revenue growth for the latest fiscal year?",
    client_id="AAPL",
    query_type="standard"
)

# Comparative analysis
result = rag_engine.query(
    question="Compare revenue growth across all companies",
    query_type="comparative",
    enable_cross_client=True
)

print(result["answer"])
print("Sources:", len(result["sources"]))
```

### 3. Report Generation

```python
from knowledge_base.src.generation.report_generator import FinancialReportGenerator, ReportConfig

generator = FinancialReportGenerator()

# Comprehensive report
config = ReportConfig(
    client_id="AAPL",
    report_type="comprehensive",
    output_format="markdown"
)

report = generator.generate_client_report("AAPL", config)
print("Report generated:", report["output_file"])

# Comparative report
config = ReportConfig(
    report_type="comparative",
    output_format="markdown"
)

report = generator.generate_comparative_report(
    client_ids=["AAPL", "MSFT", "GOOGL"],
    config=config
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: See the inline code documentation
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions and ideas

## 🔮 Roadmap

- [ ] Enhanced financial metrics extraction
- [ ] Real-time market data integration
- [ ] Advanced visualization capabilities
- [ ] Multi-language support
- [ ] Enterprise security features
- [ ] API endpoints for external integration
- [ ] Mobile application
- [ ] Advanced analytics dashboard

---
