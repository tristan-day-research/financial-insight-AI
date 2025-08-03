# Financial Knowledge Base Architecture Guide

## Overview

This document outlines the architecture decisions and patterns for the federated hybrid knowledge base designed for financial RAG applications.

## Key Architecture Decisions

### 1. SQL Database Design: Single Multi-Tenant Database

**Decision**: Use one centralized SQL database with client isolation rather than separate databases per company.

**Benefits**:
- Simplified management, backup, and maintenance
- Enables federated analytics across companies when needed  
- Better resource efficiency and horizontal scaling
- Row-level security implemented at application layer

**Implementation**:
```python
# All tables include client_id for isolation
class Document(Base):
    client_id = Column(String(20), ForeignKey("clients.id"), index=True)
    
class FinancialMetric(Base):
    client_id = Column(String(20), ForeignKey("clients.id"), index=True)
```

### 2. Separation of Concerns

**Document Processing Pipeline**:
1. **Extraction** (src/ingestion/): Pure data extraction from documents
2. **Storage Management** (src/storage/): Database operations and data persistence  
3. **Retrieval** (src/retrieval/): Query processing and federated search

**Benefits**:
- Clear responsibilities and easier testing
- Modular components can be developed independently
- Extensible to new document types
- Database logic centralized for consistency

## Module Structure

### Document Extraction Layer

#### Base Extractor Pattern
```python
# Base class for all extractors
class BaseFinancialExtractor(ABC):
    @abstractmethod
    def process_document(self, file_path: str, company_id: str) -> List[FinancialMetric]:
        pass
```

#### SEC-Specific Extractor
```python
# SEC filings extractor  
class SECDataExtractor(BaseFinancialExtractor):
    # Handles 10-K, 10-Q, 8-K filings
    # XBRL priority with PDF/text fallback
    # 98% confidence for XBRL, 85% for PDF tables, 75% for text patterns
```

#### Extensible Pattern
```python
# Example for future document types
class EarningsReportExtractor(BaseFinancialExtractor):
    # Earnings calls, analyst reports
    # Document-specific patterns and validation
```

### Storage Management Layer

#### SQL Manager
```python
class FinancialMetricsManager:
    def save_extracted_metrics(self, metrics: List[FinancialMetric], document_id: int)
    def get_metrics_for_client(self, client_id: str, fiscal_year: int) 
    def validate_client_metrics(self, client_id: str, fiscal_year: int)
    def get_comparative_metrics(self, client_ids: List[str], metric_names: List[str])
```

### Vector Storage Layer
```python
class FinancialVectorStore:
    # Client isolation with metadata filtering
    # Supports FAISS, ChromaDB, Pinecone
    # Cross-reference with SQL store via document IDs
```

## Usage Examples

### 1. Processing SEC Documents

```python
# Step 1: Extract financial data
from knowledge_base.src.ingestion.sec_sql_extractor import SECDataExtractor
extractor = SECDataExtractor()
metrics = extractor.process_document("10-K_2023.pdf", "AAPL")

# Step 2: Save to database
from knowledge_base.src.storage.sql_manager import FinancialMetricsManager
sql_manager = FinancialMetricsManager()
sql_manager.save_extracted_metrics(metrics, document_id=123)

# Step 3: Validate and analyze
validation = sql_manager.validate_client_metrics("AAPL", 2023)
comparative = sql_manager.get_comparative_metrics(
    ["AAPL", "MSFT"], ["revenue", "net_income"], 2023
)
```

### 2. Adding New Document Types

```python
# Create new extractor inheriting from base
class MyCustomExtractor(BaseFinancialExtractor):
    def _get_supported_extensions(self):
        return ['.xlsx', '.csv']
    
    def _get_metric_mappings(self):
        return {'total_sales': 'revenue'}
    
    def process_document(self, file_path, company_id):
        # Custom extraction logic
        pass

# Register with global registry
from knowledge_base.src.ingestion.base_extractor import extractor_registry
extractor_registry.register('custom_type', MyCustomExtractor())
```

### 3. Client Isolation Example

```python
# Each client only sees their own data
apple_metrics = sql_manager.get_metrics_for_client("AAPL", 2023)
microsoft_metrics = sql_manager.get_metrics_for_client("MSFT", 2023)

# Federated analysis (when authorized)
tech_comparison = sql_manager.get_comparative_metrics(
    ["AAPL", "MSFT", "GOOGL"], 
    ["revenue", "net_income"], 
    2023
)
```

## Security Considerations

### Client Data Isolation
- All database queries filtered by `client_id`
- Vector store metadata includes client isolation
- Application-level access controls

### Data Validation
- Financial metrics validation (accounting equation checks)
- Confidence scoring for extracted values
- Source tracking and audit trails

## Extension Points

### 1. New Document Types
- Inherit from `BaseFinancialExtractor`
- Implement document-specific patterns
- Register with `ExtractorRegistry`

### 2. New Metrics
- Add to metric mappings in extractors
- Update validation rules if needed
- Extend SQL schema if necessary

### 3. New Storage Backends
- Vector stores: Implement `BaseVectorStore` interface
- SQL: Extend `FinancialSQLStore` with new tables

## Performance Considerations

### Extraction Pipeline
- XBRL processing: ~2-3 seconds per document
- PDF processing: ~10-15 seconds per document  
- Batch processing recommended for multiple documents

### Database Optimization
- Indexes on `client_id`, `fiscal_year`, `metric_name`
- Partitioning by client for large datasets
- Connection pooling for concurrent access

### Vector Search
- Embedding caching for repeated queries
- Client-specific index sharding for scale
- Hybrid search combining SQL filters + vector similarity

## Migration Strategy

For existing implementations:
1. Move database setup from extractors to `sql_manager.py`
2. Update imports to use renamed modules
3. Implement base extractor inheritance
4. Test client isolation functionality
5. Add new document types as needed

This architecture provides a solid foundation for scaling to multiple document types while maintaining security and performance.