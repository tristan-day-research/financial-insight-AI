# Testing Strategy for Financial Insight AI

This document outlines the comprehensive testing strategy for the Financial Insight AI project, covering unit tests, integration tests, and performance tests.

## 🏗️ Testing Architecture

### Test Structure
```
knowledge_base/tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_sec_downloader.py   # SEC downloader tests
├── test_sql_storage.py      # SQL storage tests
├── test_document_processing.py # Document processing tests
├── test_rag_engine.py       # RAG engine tests
├── test_integration.py      # End-to-end integration tests
└── README.md               # This file
```

## 🧪 Test Categories

### 1. Unit Tests
- **Purpose**: Test individual components in isolation
- **Coverage**: Each class and method has dedicated tests
- **Mocking**: External dependencies are mocked
- **Speed**: Fast execution (< 1 second per test)

**Examples:**
- `TestSECDownloader`: Tests SEC API client functionality
- `TestSQLStore`: Tests database operations
- `TestFinancialDocumentProcessor`: Tests document chunking
- `TestRAGEngine`: Tests search and response generation

### 2. Integration Tests
- **Purpose**: Test component interactions and data flow
- **Coverage**: Complete pipeline from download to query
- **Real Data**: Uses temporary databases and files
- **Speed**: Medium execution (1-5 seconds per test)

**Examples:**
- `TestCompletePipeline`: End-to-end data processing
- `TestDataConsistency`: Verify data integrity across components
- `TestErrorHandling`: Test error scenarios

### 3. Performance Tests
- **Purpose**: Ensure system performance under load
- **Coverage**: Response times and resource usage
- **Benchmarks**: Performance thresholds and SLAs
- **Speed**: Variable execution (5-30 seconds per test)

**Examples:**
- `TestRAGEnginePerformance`: Search and response generation speed
- `TestPipelinePerformance`: Complete pipeline throughput

## 🚀 Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest knowledge_base/tests/test_sec_downloader.py

# Run specific test class
pytest knowledge_base/tests/test_sec_downloader.py::TestSECDownloader

# Run specific test method
pytest knowledge_base/tests/test_sec_downloader.py::TestSECDownloader::test_initialization
```

### Test Categories
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run performance tests
pytest -m performance

# Run tests excluding slow ones
pytest -m "not slow"

# Run tests with coverage
pytest --cov=knowledge_base --cov-report=html
```

### Test Configuration
```bash
# Run tests with specific settings
pytest --tb=short --disable-warnings

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto

# Generate coverage report
pytest --cov=knowledge_base --cov-report=term-missing --cov-report=html
```

## 📊 Test Coverage

### Coverage Targets
- **Overall Coverage**: ≥ 70%
- **Critical Paths**: ≥ 90%
- **Error Handling**: ≥ 80%
- **Integration Points**: ≥ 85%

### Coverage Areas
1. **Data Ingestion** (SEC downloads, document processing)
2. **Data Storage** (SQL operations, vector storage)
3. **Data Extraction** (Financial metrics, text parsing)
4. **Query Processing** (RAG engine, search functionality)
5. **Error Handling** (API failures, database errors)

## 🔧 Test Fixtures

### Common Fixtures
- `test_settings`: Test configuration with temporary paths
- `temp_db`: Temporary SQLite database for testing
- `mock_sec_response`: Mock SEC API responses
- `sample_sec_filing`: Sample SEC filing data
- `sample_financial_metrics`: Sample financial metrics
- `mock_downloader`: Mock SEC downloader
- `sample_document_chunk`: Sample document chunk

### Usage Example
```python
def test_my_function(temp_db, sample_sec_filing):
    """Test using fixtures."""
    # Use temp_db for database operations
    # Use sample_sec_filing for test data
    result = my_function(sample_sec_filing)
    assert result is not None
```

## 🐛 Debugging Tests

### Common Issues
1. **Import Errors**: Ensure project root is in Python path
2. **Database Lock**: Use temporary databases for tests
3. **File Permissions**: Use temporary directories
4. **API Rate Limits**: Mock external API calls

### Debug Commands
```bash
# Run with debug output
pytest -v -s

# Run single test with debugger
pytest -v -s --pdb

# Run with detailed error info
pytest -v --tb=long

# Check test discovery
pytest --collect-only
```

## 📈 Continuous Integration

### GitHub Actions Workflow
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=knowledge_base --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 🎯 Best Practices

### Test Design
1. **Arrange-Act-Assert**: Structure tests clearly
2. **Single Responsibility**: Each test has one purpose
3. **Descriptive Names**: Use clear test method names
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Execution**: Keep tests under 1 second when possible

### Mocking Strategy
1. **External APIs**: Always mock SEC API calls
2. **File System**: Use temporary files/directories
3. **Database**: Use in-memory SQLite for tests
4. **LLM Calls**: Mock OpenAI API responses

### Data Management
1. **Test Data**: Use realistic but minimal test data
2. **Cleanup**: Always clean up temporary resources
3. **Isolation**: Each test should be independent
4. **Fixtures**: Reuse common test data via fixtures

## 🔍 Test Monitoring

### Metrics to Track
- **Test Coverage**: Percentage of code covered
- **Test Execution Time**: Total time for test suite
- **Test Reliability**: Flaky test detection
- **Test Distribution**: Unit vs integration test ratio

### Reporting
- **Coverage Reports**: HTML and XML coverage reports
- **Test Results**: Pass/fail statistics
- **Performance Metrics**: Response time benchmarks
- **Error Analysis**: Common failure patterns

## 🚨 Error Handling Tests

### Critical Error Scenarios
1. **API Failures**: SEC API unavailable
2. **Database Errors**: Connection failures, constraint violations
3. **File System Errors**: Disk full, permission denied
4. **Network Errors**: Timeout, connection refused
5. **Data Validation**: Invalid input data

### Error Test Patterns
```python
def test_error_handling():
    """Test error handling patterns."""
    # Test with invalid input
    with pytest.raises(ValueError):
        function_with_invalid_input()
    
    # Test with missing data
    result = function_with_missing_data()
    assert result is None
    
    # Test with network error
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.RequestException()
        result = function_with_network_call()
        assert result is None
```

## 📚 Additional Resources

### Documentation
- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

### Tools
- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-xdist**: Parallel test execution

### Configuration Files
- `pytest.ini`: Pytest configuration
- `conftest.py`: Shared fixtures and configuration
- `.coveragerc`: Coverage configuration (if needed) 