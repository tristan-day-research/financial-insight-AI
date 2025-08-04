"""
Vector Database Text Processor
Handles text chunking, embedding, and storage for the financial RAG system
"""

import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import sqlite3
from datetime import datetime

@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    id: str
    text: str
    company_id: str
    source_document: str
    chunk_index: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    doc_type: Optional[str] = None
    period: Optional[str] = None
    metadata: Optional[Dict] = None

class VectorProcessor:
    """Process documents into vector embeddings with rich metadata"""
    
    def __init__(self, 
                 vector_db_path: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings(allow_reset=True)
        )
        
        # Financial term patterns for enhanced metadata
        self.financial_terms = {
            'revenue_terms': ['revenue', 'sales', 'income', 'earnings'],
            'expense_terms': ['cost', 'expense', 'expenditure', 'outlay'],
            'balance_terms': ['assets', 'liabilities', 'equity', 'balance'],
            'cash_terms': ['cash', 'liquidity', 'flow', 'working capital'],
            'growth_terms': ['growth', 'increase', 'decrease', 'change'],
            'performance_terms': ['margin', 'ratio', 'return', 'yield']
        }

    def get_or_create_collection(self, company_id: str):
        """Get or create a collection for a specific company"""
        collection_name = f"{company_id}_chunks"
        
        try:
            collection = self.client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except ValueError:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"company_id": company_id}
            )
            print(f"Created new collection: {collection_name}")
            
        return collection

    def process_document(self, 
                        file_path: str, 
                        company_id: str,
                        doc_type: Optional[str] = None,
                        period: Optional[str] = None) -> List[TextChunk]:
        """Process a document into text chunks with metadata"""
        
        file_path = Path(file_path)
        print(f"Processing {file_path.name} for vector database")
        
        # Extract text from document
        if file_path.suffix.lower() == '.pdf':
            text_content, page_info = self._extract_text_from_pdf(file_path)
        else:
            text_content = self._extract_text_from_file(file_path)
            page_info = {}
        
        # Create text chunks
        chunks = self._create_text_chunks(
            text=text_content,
            company_id=company_id,
            source_document=file_path.name,
            doc_type=doc_type,
            period=period,
            page_info=page_info
        )
        
        print(f"Created {len(chunks)} text chunks")
        return chunks

    def _extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract text from PDF with page information"""
        try:
            import pdfplumber
            
            full_text = []
            page_info = {}  # Map text positions to page numbers
            current_position = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    full_text.append(page_text)
                    
                    # Track which part of text corresponds to which page
                    page_info[current_position] = page_num
                    current_position += len(page_text) + 1  # +1 for newline
                    
            return '\n'.join(full_text), page_info
            
        except ImportError:
            print("pdfplumber not available, falling back to basic text extraction")
            return self._extract_text_from_file(pdf_path), {}

    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {file_path}")

    def _create_text_chunks(self, 
                           text: str,
                           company_id: str,
                           source_document: str,
                           doc_type: Optional[str],
                           period: Optional[str],
                           page_info: Dict) -> List[TextChunk]:
        """Split text into overlapping chunks with rich metadata"""
        
        chunks = []
        
        # Simple sentence-aware chunking
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_sentences = []
        chunk_index = 0
        text_position = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
            else:
                # Create chunk from current content
                if current_chunk.strip():
                    chunk = self._create_chunk_object(
                        text=current_chunk.strip(),
                        company_id=company_id,
                        source_document=source_document,
                        chunk_index=chunk_index,
                        doc_type=doc_type,
                        period=period,
                        text_position=text_position,
                        page_info=page_info
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) > 1 else []
                current_chunk = " ".join(overlap_sentences + [sentence])
                current_sentences = overlap_sentences + [sentence]
            
            text_position += len(sentence) + 1
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = self._create_chunk_object(
                text=current_chunk.strip(),
                company_id=company_id,
                source_document=source_document,
                chunk_index=chunk_index,
                doc_type=doc_type,
                period=period,
                text_position=text_position,
                page_info=page_info
            )
            chunks.append(chunk)
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving financial formatting"""
        
        # Pre-process to protect financial abbreviations
        text = re.sub(r'\b(Mr|Mrs|Dr|Inc|Corp|Ltd|LLC)\.', r'\1<DOT>', text)
        text = re.sub(r'\$(\d+)\.(\d+)', r'$\1<DOT>\2', text)  # Protect dollar amounts
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk_object(self,
                            text: str,
                            company_id: str,
                            source_document: str,
                            chunk_index: int,
                            doc_type: Optional[str],
                            period: Optional[str],
                            text_position: int,
                            page_info: Dict) -> TextChunk:
        """Create a TextChunk object with rich metadata"""
        
        # Generate unique chunk ID
        chunk_id = self._generate_chunk_id(company_id, source_document, chunk_index)
        
        # Determine page number
        page_number = self._get_page_for_position(text_position, page_info)
        
        # Detect section from text content
        section = self._detect_section(text)
        
        # Analyze financial content
        financial_analysis = self._analyze_financial_content(text)
        
        # Create enhanced metadata
        metadata = {
            "company_id": company_id,
            "source_document": source_document,
            "doc_type": doc_type or self._infer_doc_type(source_document),
            "period": period or "unknown",
            "section": section,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "word_count": len(text.split()),
            "char_count": len(text),
            
            # Financial content analysis
            "contains_financial_data": financial_analysis["has_financial_data"],
            "financial_term_density": financial_analysis["term_density"],
            "dominant_financial_theme": financial_analysis["dominant_theme"],
            "contains_numbers": bool(re.search(r'\$[\d,]+', text)),
            
            # Processing metadata
            "processed_at": datetime.now().isoformat(),
            "embedding_model": "all-MiniLM-L6-v2"
        }
        
        return TextChunk(
            id=chunk_id,
            text=text,
            company_id=company_id,
            source_document=source_document,
            chunk_index=chunk_index,
            page_number=page_number,
            section=section,
            doc_type=doc_type,
            period=period,
            metadata=metadata
        )

    def _generate_chunk_id(self, company_id: str, source_document: str, chunk_index: int) -> str:
        """Generate unique, deterministic chunk ID"""
        base_name = Path(source_document).stem
        return f"{company_id}_{base_name}_chunk_{chunk_index:03d}"

    def _get_page_for_position(self, text_position: int, page_info: Dict) -> Optional[int]:
        """Determine which page a text position corresponds to"""
        if not page_info:
            return None
            
        # Find the latest page start that's before our position
        relevant_positions = [pos for pos in page_info.keys() if pos <= text_position]
        if relevant_positions:
            latest_position = max(relevant_positions)
            return page_info[latest_position]
        
        return None

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect what section of a financial document this chunk is from"""
        text_lower = text.lower()
        
        section_patterns = {
            "executive_summary": ["executive summary", "highlights", "key points"],
            "revenue_analysis": ["revenue", "sales", "net sales", "income statement"],
            "balance_sheet": ["balance sheet", "assets", "liabilities", "equity"],
            "cash_flow": ["cash flow", "operating activities", "investing activities"],
            "risk_factors": ["risk factors", "risks", "uncertainties"],
            "md_a": ["management discussion", "md&a", "analysis of financial condition"],
            "notes": ["notes to", "note ", "accounting policies"],
            "governance": ["corporate governance", "board of directors", "committees"]
        }
        
        for section_name, keywords in section_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return section_name
                
        return "general"

    def _analyze_financial_content(self, text: str) -> Dict:
        """Analyze the financial content of a text chunk"""
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Count financial terms by category
        term_counts = {}
        total_financial_terms = 0
        
        for category, terms in self.financial_terms.items():
            count = sum(text_lower.count(term) for term in terms)
            term_counts[category] = count
            total_financial_terms += count
        
        # Calculate term density
        term_density = total_financial_terms / word_count if word_count > 0 else 0
        
        # Find dominant theme
        dominant_theme = max(term_counts.items(), key=lambda x: x[1])[0] if total_financial_terms > 0 else "general"
        
        # Check for financial data indicators
        has_financial_data = (
            term_density > 0.05 or  # >5% financial terms
            bool(re.search(r'\$[\d,]+', text)) or  # Contains dollar amounts
            bool(re.search(r'\d+%', text)) or      # Contains percentages
            bool(re.search(r'\d+\.?\d*\s*(million|billion|thousand)', text, re.IGNORECASE))
        )
        
        return {
            "has_financial_data": has_financial_data,
            "term_density": round(term_density, 3),
            "dominant_theme": dominant_theme,
            "term_counts": term_counts
        }

    def _infer_doc_type(self, filename: str) -> str:
        """Infer document type from filename"""
        filename_lower = filename.lower()
        
        if '10-k' in filename_lower:
            return '10-K'
        elif '10-q' in filename_lower:
            return '10-Q'
        elif '8-k' in filename_lower:
            return '8-K'
        elif 'earnings' in filename_lower:
            return 'earnings_transcript'
        elif 'proxy' in filename_lower:
            return 'proxy_statement'
        else:
            return 'unknown'

    def store_chunks(self, chunks: List[TextChunk]) -> int:
        """Store text chunks in vector database"""
        if not chunks:
            return 0
        
        company_id = chunks[0].company_id
        collection = self.get_or_create_collection(company_id)
        
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Store in ChromaDB
        try:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"Successfully stored {len(chunks)} chunks for {company_id}")
            return len(chunks)
            
        except Exception as e:
            print(f"Error storing chunks: {e}")
            return 0

    def search_chunks(self, 
                     company_id: str,
                     query: str,
                     n_results: int = 5,
                     filters: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant chunks"""
        
        collection = self.get_or_create_collection(company_id)
        
        # Build where clause for filtering
        where_clause = {}
        if filters:
            where_clause.update(filters)
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []

    def get_collection_stats(self, company_id: str) -> Dict:
        """Get statistics about a company's chunk collection"""
        collection = self.get_or_create_collection(company_id)
        
        try:
            count = collection.count()
            
            # Get sample of metadata for analysis
            sample_results = collection.get(limit=min(100, count))
            
            if sample_results['metadatas']:
                # Analyze document types
                doc_types = {}
                sections = {}
                periods = {}
                
                for metadata in sample_results['metadatas']:
                    doc_type = metadata.get('doc_type', 'unknown')
                    section = metadata.get('section', 'unknown')
                    period = metadata.get('period', 'unknown')
                    
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    sections[section] = sections.get(section, 0) + 1
                    periods[period] = periods.get(period, 0) + 1
                
                return {
                    'total_chunks': count,
                    'document_types': doc_types,
                    'sections': sections,
                    'periods': periods
                }
            
            return {'total_chunks': count}
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def main():
    """Example usage"""
    processor = VectorProcessor()
    
    # Process a sample document
    sample_file = "data/raw_documents/apple/10-K_2023.pdf"
    
    if Path(sample_file).exists():
        # Process document into chunks
        chunks = processor.process_document(
            file_path=sample_file,
            company_id="apple",
            doc_type="10-K",
            period="FY_2023"
        )
        
        # Store chunks in vector database
        stored_count = processor.store_chunks(chunks)
        print(f"Stored {stored_count} chunks")
        
        # Test search
        search_results = processor.search_chunks(
            company_id="apple",
            query="iPhone revenue growth",
            n_results=3
        )
        
        print("\nSearch Results:")
        for result in search_results:
            print(f"- {result['text'][:100]}... (score: {result['distance']:.3f})")
        
        # Get collection statistics
        stats = processor.get_collection_stats("apple")
        print(f"\nCollection Stats: {stats}")
        
    else:
        print(f"Sample file {sample_file} not found")


if __name__ == "__main__":
    main()