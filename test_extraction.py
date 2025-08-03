#!/usr/bin/env python3

from knowledge_base.src.ingestion.sql_extractor import SECDataExtractor
from pathlib import Path

# Initialize extractor
extractor = SECDataExtractor()

# Test file path
file_path = Path('/Users/daylight/Desktop/Financial Insight AI/knowledge_base/data/raw/RDDT/10-Q/000171344525000196/000171344525000196.html')

# Extract metrics
metrics = extractor.process_document(str(file_path), 'RDDT')

print(f'Extracted {len(metrics)} metrics:')
for metric in metrics:
    print(f'- {metric.metric_name}: ${metric.value:,.2f} ({metric.extraction_method}, confidence: {metric.confidence})')
    print(f'  Period: {metric.period}, Unit: {metric.unit}')
    print(f'  Raw text: {metric.raw_text[:100]}...')
    print() 