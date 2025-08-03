#!/usr/bin/env python3
"""
Simple Database Analysis Script
Tests the financial_kb.db database and shows its current state.
"""

import sqlite3
import pandas as pd
from pathlib import Path

def analyze_database():
    """Analyze the financial knowledge base database."""
    
    # Database path
    db_path = Path("knowledge_base/data/financial_kb.db")
    
    print("=== FINANCIAL KNOWLEDGE BASE DATABASE ANALYSIS ===")
    print(f"Database path: {db_path}")
    print(f"Database exists: {db_path.exists()}")
    
    if not db_path.exists():
        print("❌ Database not found!")
        return
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # Get all tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql_query(tables_query, conn)
        print(f"\n📋 Tables found: {tables['name'].tolist()}")
        
        # Analyze each table
        for table_name in tables['name']:
            print(f"\n--- {table_name.upper()} TABLE ---")
            
            # Get table schema
            schema_query = f"PRAGMA table_info({table_name})"
            schema = pd.read_sql_query(schema_query, conn)
            print(f"Columns: {schema['name'].tolist()}")
            
            # Get record count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count = pd.read_sql_query(count_query, conn)
            record_count = count['count'].iloc[0]
            print(f"Records: {record_count}")
            
            # Show sample data if records exist
            if record_count > 0:
                sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                sample = pd.read_sql_query(sample_query, conn)
                print("Sample data:")
                print(sample.to_string(index=False))
            else:
                print("(No data)")
        
        # Additional analysis
        print(f"\n=== ADDITIONAL ANALYSIS ===")
        
        # Check for any financial metrics
        metrics_query = "SELECT COUNT(*) as count FROM financial_metrics"
        metrics_count = pd.read_sql_query(metrics_query, conn)['count'].iloc[0]
        print(f"Financial metrics extracted: {metrics_count}")
        
        # Check for any clients
        clients_query = "SELECT COUNT(*) as count FROM clients"
        clients_count = pd.read_sql_query(clients_query, conn)['count'].iloc[0]
        print(f"Clients registered: {clients_count}")
        
        # Check document processing status
        docs_query = """
        SELECT 
            client_id,
            filing_type,
            filing_date,
            has_revenue_data,
            has_profit_data,
            has_balance_sheet,
            has_cash_flow,
            financial_density
        FROM documents
        """
        docs = pd.read_sql_query(docs_query, conn)
        if not docs.empty:
            print(f"\nDocument processing status:")
            print(docs.to_string(index=False))
        
        # Check for any chunks
        chunks_query = "SELECT COUNT(*) as count FROM document_chunks"
        chunks_count = pd.read_sql_query(chunks_query, conn)['count'].iloc[0]
        print(f"Document chunks created: {chunks_count}")
        
        print(f"\n✅ Database analysis complete!")
        
    except Exception as e:
        print(f"❌ Error analyzing database: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_database() 