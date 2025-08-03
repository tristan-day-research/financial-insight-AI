"""
Financial document processor for extracting and chunking text from various file formats.
Handles PDF, DOCX, HTML, and text files with financial content optimization.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import re
import json
from datetime import datetime
import hashlib
import pandas as pd

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from bs4 import BeautifulSoup


# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_base.config.settings import get_settings

logger = logging.getLogger(__name__)


class FinancialDocumentProcessor:
    """Processes financial documents with optimized chunking and metadata extraction."""

    def __init__(self):
        self.settings = get_settings()
        # Enhanced text splitter with financial-aware separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.data.chunk_size,
            chunk_overlap=self.settings.data.chunk_overlap,
            separators=[
                "\n\n\n",  # Document sections
                "\n\n",    # Paragraphs
                "\.\s+",   # Sentences (with space after period)
                "\n",      # Lines
                " ",       # Words
                ""         # Characters
            ],
            keep_separator=True
        )
        
        # Enhanced financial section patterns
        self.financial_section_patterns = [
            (r"PART\s+[IVX]+", "SEC Part"),  # SEC filing parts
            (r"Item\s+\d+[A-Z]?", "SEC Item"),  # SEC filing items
            (r"CONSOLIDATED\s+STATEMENTS?", "Financial Statements"),
            (r"NOTES?\s+TO\s+.*FINANCIAL\s+STATEMENTS?", "Financial Notes"),
            (r"MANAGEMENT['\s]*S\s+DISCUSSION", "MD&A"),
            (r"RISK\s+FACTORS", "Risk Factors"),
            (r"BUSINESS\s+OVERVIEW", "Business Description"),
            (r"TABLE\s+OF\s+CONTENTS", "TOC"),  # For table preservation
            (r"\bEXHIBIT\s+\d+", "Exhibit")  # For exhibits
        ]


    def process_sec_filing(self, file_path: str, metadata: Dict) -> List[Document]:
        """
        Process a SEC filing document into chunks with financial context.
        
        Args:
            file_path: Path to the SEC filing
            metadata: Document metadata from downloader
            
        Returns:
            List of LangChain Document objects with enhanced metadata
        """
        try:
            # Read and clean the document
            raw_text = self._read_sec_file(file_path)
            cleaned_text = self._clean_sec_text(raw_text)
            
            # Extract document structure
            sections = self._extract_sec_sections(cleaned_text)
            
            # Create chunks with section-aware splitting
            chunks = []
            for section_name, section_text in sections.items():
                section_chunks = self._create_financial_chunks(
                    section_text, 
                    section_name, 
                    metadata
                )
                chunks.extend(section_chunks)
            
            # If no sections found, process as whole document
            if not chunks:
                chunks = self._create_financial_chunks(cleaned_text, "full_document", metadata)
            
            logger.info(f"Processed {file_path}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []

    def _read_sec_file(self, file_path: str) -> str:
        """Enhanced file reader with Unstructured/PyPDFLoader support"""
        if str(file_path).lower().endswith('.pdf'):
            # Option 1: Use Unstructured for better table handling
            try:
                elements = partition_pdf(filename=file_path, strategy="hi_res")
                return "\n\n".join([str(el) for el in elements])
            except Exception as e:
                logger.warning(f"Error reading PDF {file_path}: {str(e)}")
                # Fall back to text reading if PDF fails
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            # Option 2: Use PyPDFLoader (simpler but less table-aware)
            # loader = PyPDFLoader(file_path)
            # return "\n\n".join([page.page_content for page in loader.load()])
        else:
            # Existing text file handling
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read with errors ignored
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()


    def _clean_sec_text(self, text: str) -> str:
        """Clean SEC document text, removing HTML, formatting artifacts."""
        # Remove HTML tags if present
        
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove SEC document headers and footers
        text = re.sub(r'<SEC-DOCUMENT>.*?</SEC-DOCUMENT>', '', text, flags=re.DOTALL)
        text = re.sub(r'<DOCUMENT>.*?</DOCUMENT>', '', text, flags=re.DOTALL)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Page\s+\d+.*$', '', text, flags=re.MULTILINE)
        
        # Clean financial number formatting artifacts
        text = re.sub(r'\$\s+(\d)', r'$\1', text)  # Fix spaced dollar signs
        text = re.sub(r'(\d)\s+,\s+(\d)', r'\1,\2', text)  # Fix spaced commas in numbers
        
        return text.strip()
    
    def _extract_sec_sections(self, text: str) -> Dict[str, str]:
        """Extract major sections from SEC filing."""
        sections = {}
        
        # Try to identify sections using common patterns
        for pattern, section_type in self.financial_section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for i, match in enumerate(matches):
                section_start = match.start()
                section_name = f"{section_type}: {match.group().strip()}"
                
                # Find the end of this section (start of next section or end of document)
                if i + 1 < len(matches):
                    section_end = matches[i + 1].start()
                else:
                    # Look for next major section or end of document
                    next_section = self._find_next_major_section(text, section_start)
                    section_end = next_section if next_section else len(text)
                
                section_text = text[section_start:section_end].strip()
                
                if len(section_text) > 100:  # Only include substantial sections
                    sections[section_name] = section_text
        
        # If no sections found, try basic PART/Item extraction
        if not sections:
            sections = self._extract_basic_sections(text)
        
        return sections
    
    def _find_next_major_section(self, text: str, current_pos: int) -> Optional[int]:
        """Find the position of the next major section."""
        remaining_text = text[current_pos + 100:]  # Skip current section header
        
        for pattern, _ in self.financial_section_patterns:
            match = re.search(pattern, remaining_text, re.IGNORECASE)
            if match:
                return current_pos + 100 + match.start()
        
        return None
    
    def _extract_basic_sections(self, text: str) -> Dict[str, str]:
        """Basic section extraction as fallback."""
        sections = {}
        
        # Split on common section dividers
        parts = re.split(r'\n\s*(?=PART\s+[IVX]+|Item\s+\d+)', text, flags=re.IGNORECASE)
        
        for i, part in enumerate(parts):
            if len(part.strip()) > 200:  # Only include substantial parts
                # Extract section name from first line
                lines = part.strip().split('\n')
                section_name = lines[0][:50] if lines else f"Section_{i}"
                sections[section_name] = part.strip()
        
        return sections if sections else {"full_document": text}
    
    def _create_financial_chunks(
        self, 
        text: str, 
        section_name: str, 
        base_metadata: Dict
    ) -> List[Document]:
        """Create optimized chunks for financial content."""
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(text_chunks):
            # Enhanced metadata for each chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "section": section_name,
                "chunk_index": i,
                "chunk_size": len(chunk),
                "processed_date": datetime.now().isoformat(),
            })
            
            # Add financial content analysis
            financial_analysis = self._analyze_financial_content(chunk)
            chunk_metadata.update(financial_analysis)
            
            # Create LangChain Document
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def _analyze_financial_content(self, text: str) -> Dict:
        """Analyze chunk for financial content and context."""
        analysis = {
            "content_type": "general",
            "financial_density": 0.0,
            "contains_numbers": False,
            "contains_financial_terms": False,
            "key_topics": []
        }
        
        text_lower = text.lower()
        
        # Count financial keywords
        financial_keywords = self.settings.data.financial_keywords
        keyword_matches = sum(1 for keyword in financial_keywords if keyword.lower() in text_lower)
        
        # Calculate financial density (keywords per 100 words)
        word_count = len(text.split())
        if word_count > 0:
            analysis["financial_density"] = (keyword_matches / word_count) * 100
        
        # Check for numbers (potential financial data)
        analysis["contains_numbers"] = bool(re.search(r'\$[\d,]+|\d+\.\d+%|\d{1,3}(?:,\d{3})+', text))
        
        # Check for financial terms
        analysis["contains_financial_terms"] = keyword_matches > 0
        
        # Determine content type based on patterns
        if any(term in text_lower for term in ["balance sheet", "consolidated statements"]):
            analysis["content_type"] = "financial_statements"
        elif any(term in text_lower for term in ["risk factors", "risks"]):
            analysis["content_type"] = "risk_analysis"
        elif any(term in text_lower for term in ["management discussion", "md&a"]):
            analysis["content_type"] = "management_analysis"
        elif any(term in text_lower for term in ["business overview", "description of business"]):
            analysis["content_type"] = "business_description"
        elif analysis["financial_density"] > 2.0:
            analysis["content_type"] = "financial_data"
        
        # Extract key topics (simplified)
        key_topics = []
        if "revenue" in text_lower:
            key_topics.append("revenue")
        if any(term in text_lower for term in ["profit", "income", "earnings"]):
            key_topics.append("profitability")
        if any(term in text_lower for term in ["debt", "liabilities"]):
            key_topics.append("debt")
        if "cash flow" in text_lower:
            key_topics.append("cash_flow")
        
        analysis["key_topics"] = key_topics
        
        return analysis
    
    def process_multiple_documents(self, document_list: List[Dict]) -> List[Document]:
        """Process multiple documents in batch."""
        all_chunks = []
        
        for doc_metadata in document_list:
            file_path = doc_metadata.get("file_path")
            if not file_path or not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            chunks = self.process_sec_filing(file_path, doc_metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(document_list)} documents into {len(all_chunks)} total chunks")
        return all_chunks
    
    def save_processed_chunks(self, chunks: List[Document], output_file: str):
        """Save processed chunks to file for inspection/debugging."""
        output_path = Path(self.settings.data.processed_data_path) / output_file
        
        # Convert to DataFrame for easy saving
        chunk_data = []
        for chunk in chunks:
            chunk_info = {
                "content": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
                "full_content_length": len(chunk.page_content),
                **chunk.metadata
            }
            chunk_data.append(chunk_info)
        
        df = pd.DataFrame(chunk_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    """Example usage of document processor."""
    processor = FinancialDocumentProcessor()
    
    # Example: Process a sample document
    sample_metadata = {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "client_id": "AAPL",
        "filing_date": "2023-10-01"
    }
    
    # This would process an actual SEC filing
    print("Document processor initialized successfully")
    print(f"Chunk size: {processor.settings.data.chunk_size}")
    print(f"Financial keywords: {len(processor.settings.data.financial_keywords)}")


if __name__ == "__main__":
    main()