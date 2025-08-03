#!/usr/bin/env python3

import requests
import json
from knowledge_base.config.settings import get_settings

def investigate_sec_api():
    settings = get_settings()
    
    # Headers for SEC API
    headers = {
        "User-Agent": f"{settings.api.sec_user_agent} {settings.api.sec_email}",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    }
    
    # Test with multiple companies
    companies = [
        ("MSTR", "0001050446"),
        ("AAPL", "0000320193"),
        ("MSFT", "0000789019"),
        ("GOOGL", "0001652044")
    ]
    
    for ticker, cik in companies:
        print(f"\n{'='*50}")
        print(f"Investigating {ticker} (CIK: {cik})")
        print(f"{'='*50}")
        
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            continue
        
        data = response.json()
        filings = data['filings']['recent']
        
        # Look for 10-K filings specifically
        if 'form' in filings:
            form_indices = [i for i, form in enumerate(filings['form']) if form == '10-K']
            if form_indices:
                print(f"Found {len(form_indices)} 10-K filings")
                
                # Look at the most recent 10-K
                idx = form_indices[0]
                accession = filings['accessionNumber'][idx].replace('-', '')
                report_date = filings['reportDate'][idx]
                
                print(f"Most recent 10-K:")
                print(f"  Accession: {accession}")
                print(f"  Report Date: {report_date}")
                
                # Try to get the actual filing to look for fiscal year end
                try:
                    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
                    index_response = requests.get(index_url, headers=headers)
                    
                    if index_response.status_code == 200:
                        index_data = index_response.json()
                        files = index_data.get('directory', {}).get('item', [])
                        
                        if isinstance(files, dict):
                            files = [files]
                        
                        # Look for the main filing document
                        main_doc = None
                        for file in files:
                            name = file.get('name', '').lower()
                            if name.endswith('.txt'):
                                main_doc = file['name']
                                break
                        
                        if main_doc:
                            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{main_doc}"
                            doc_response = requests.get(doc_url, headers=headers)
                            
                            if doc_response.status_code == 200:
                                content = doc_response.text[:5000]  # First 5000 chars
                                
                                # Look for fiscal year end patterns
                                fiscal_patterns = [
                                    "FISCAL YEAR END:",
                                    "FISCAL YEAR END",
                                    "FISCAL YEAR:",
                                    "FISCAL YEAR",
                                    "YEAR END:",
                                    "YEAR END"
                                ]
                                
                                found_fiscal = False
                                for pattern in fiscal_patterns:
                                    if pattern in content:
                                        # Extract the line containing fiscal year end
                                        lines = content.split('\n')
                                        for line in lines:
                                            if pattern in line:
                                                print(f"  Found fiscal year end: {line.strip()}")
                                                found_fiscal = True
                                                break
                                        if found_fiscal:
                                            break
                                
                                if not found_fiscal:
                                    print("  No fiscal year end found in document header")
                                    
                                    # Try to infer from report date
                                    if report_date:
                                        try:
                                            from datetime import datetime
                                            date_obj = datetime.strptime(report_date, '%Y-%m-%d')
                                            
                                            # Most companies have fiscal year end in September (09-30) or December (12-31)
                                            if date_obj.month >= 10:  # October or later
                                                fiscal_year = date_obj.year
                                            else:
                                                fiscal_year = date_obj.year - 1
                                            
                                            # Common fiscal year ends
                                            fiscal_year_ends = [
                                                f"{fiscal_year}-09-30",  # September 30
                                                f"{fiscal_year}-12-31",  # December 31
                                                f"{fiscal_year}-03-31",  # March 31
                                                f"{fiscal_year}-06-30"   # June 30
                                            ]
                                            
                                            print(f"  Inferred fiscal year ends: {fiscal_year_ends}")
                                            
                                        except Exception as e:
                                            print(f"  Error parsing date {report_date}: {e}")
                
                except Exception as e:
                    print(f"  Error accessing filing: {e}")
            else:
                print("No 10-K filings found")

if __name__ == "__main__":
    investigate_sec_api() 