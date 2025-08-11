import re
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup, NavigableString, Tag
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from knowledge_base.config.settings import get_settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes SEC documents with intelligent chunking and metadata extraction"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_path = Path(self.settings.data.raw_docs_base_path)
        self.chunk_size=self.settings.data.chunk_tokens
        self.overlap_size=self.settings.data.chunk_overlap

        self.output_path = Path(self.settings.data.processed_text_chunk_path)
        self.max_workers = self.settings.data.num_workers
        self.batch_size = self.settings.data.batch_size_local
        
        # Track processed documents to avoid duplicates
        self.processed_docs = set()
        self._load_processed_docs()
        
        # Financial terms for content analysis
        self.financial_terms = {
            "revenue": ["revenue", "sales", "net sales", "total revenue", "gross sales"],
            "profitability": ["profit", "income", "earnings", "margin", "ebitda", "operating income"],
            "cash_flow": ["cash flow", "operating cash", "free cash flow", "cash from operations"],
            "debt": ["debt", "liabilities", "borrowings", "notes payable", "credit facility"],
            "assets": ["assets", "inventory", "receivables", "property", "investments"],
            "risk": ["risk", "uncertainty", "contingency", "litigation", "regulatory"]
        }

    def _load_processed_docs(self):
        """Load list of already processed documents to avoid duplicates"""
        if not self.output_path or not self.output_path.exists():
            return
            
        # Look for existing chunk files and extract document IDs
        for chunk_file in self.output_path.glob("*_chunks.json"):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    if chunks and isinstance(chunks, list):
                        # Extract raw_doc_id from first chunk
                        raw_doc_id = chunks[0].get("metadata", {}).get("raw_doc_id")
                        if raw_doc_id:
                            self.processed_docs.add(raw_doc_id)
                            logger.debug(f"Found existing chunks for document: {raw_doc_id}")
            except Exception as e:
                logger.warning(f"Error reading existing chunk file {chunk_file}: {e}")
        
        logger.info(f"Loaded {len(self.processed_docs)} previously processed documents")

    def is_document_processed(self, metadata: Dict) -> bool:
        """Check if a document has already been processed"""
        raw_doc_id = metadata.get("raw_doc_id")
        return raw_doc_id in self.processed_docs

    def mark_document_processed(self, metadata: Dict):
        """Mark a document as processed"""
        raw_doc_id = metadata.get("raw_doc_id")
        if raw_doc_id:
            self.processed_docs.add(raw_doc_id)

    def _get_sec_chunks(self, doc_text: str, metadata: Dict) -> List[Dict]:
        """Creates structured chunks for SEC documents with semantic metadata."""
        
        # Process the document using our chunking system
        chunks = self._process_sec_document(
            text_content=doc_text,
            company_id=metadata.get("ticker", "UNKNOWN"),
            filing_type=metadata.get("type", "UNKNOWN"),
            filing_date=metadata.get("date", "UNKNOWN"),
            metadata=metadata
        )
        
        # Convert TextChunk objects to dictionaries
        chunk_dicts = []
        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_id": self.sha256_hash(chunk.text),
                "raw_doc_id": metadata.get("raw_doc_id", ""),
                "chunk_index": idx,
                "source": metadata.get("source", "SEC EDGAR"),
                "doc_type": "sec_filing",
                # SEC specific fields
                "filing_type": metadata.get("type"),
                "ticker": metadata.get("ticker"),
                "filing_date": metadata.get("date"),
                "accession_number": metadata.get("accession_number"),
                "path": metadata.get("file_path"),
                "vector_store_id": None,
                # Enhanced semantic metadata from chunking
                "section": chunk.section,
                "element_type": getattr(chunk, 'element_type', 'TextElement'),
                "parsing_method": getattr(chunk, 'parsing_method', 'semantic_chunking'),
                "timestamp": datetime.utcnow().isoformat(),
                
                # Rich metadata from our chunking system
                "contains_financial_data": chunk.metadata.get("contains_financial_data", False),
                "financial_term_density": chunk.metadata.get("financial_term_density", 0.0),
                "dominant_financial_theme": chunk.metadata.get("dominant_financial_theme", "general"),
                "contains_numbers": chunk.metadata.get("contains_numbers", False),
                "word_count": chunk.metadata.get("word_count", 0),
                "char_count": chunk.metadata.get("char_count", 0)
            }
            
            chunk_dicts.append({
                "text": chunk.text,
                "metadata": chunk_metadata
            })

        return chunk_dicts

    def _get_generic_chunks(self, doc_text: str, metadata: Dict) -> List[Dict]:
        """Creates chunks for generic documents using a standard text splitter."""
        # Placeholder for generic chunking - you'll implement this later
        chunks = [doc_text]  # Simple single chunk for now
        chunk_dicts = []
        
        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_id": self.sha256_hash(chunk),
                "raw_doc_id": metadata["raw_doc_id"],
                "chunk_index": idx,
                "source": metadata.get("source"),
                "doc_type": "generic",
                "title": metadata.get("title"),
                "author": metadata.get("author"),
                "creation_date": metadata.get("creation_date"),
                "path": metadata.get("file_path"),
                "vector_store_id": None,
            }
            chunk_dicts.append({"text": chunk, "metadata": chunk_metadata})
        return chunk_dicts

    def load_metadata(self, doc_folder: Path) -> Dict:
        """Load metadata from document folder"""
        metadata_file = doc_folder / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def read_text_file(self, file_path: Path) -> str:
        """Read text content from file with encoding fallback"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {file_path}")

    def process_document_folder(self, doc_folder: Path) -> List[Dict]:
        """
        Processes a single document folder by dispatching to the correct
        chunking method based on the document's metadata.
        """
        try:
            metadata = self.load_metadata(doc_folder)
            
            # Check if already processed
            if self.is_document_processed(metadata):
                logger.info(f"Skipping already processed document: {metadata.get('raw_doc_id', 'unknown')}")
                return []

            # SEC document logic: only process .html in the folder
            if "accession_number" in metadata:
                html_files = list(doc_folder.glob("*.html"))
                if not html_files:
                    logger.warning(f"No .html file found in SEC doc folder: {doc_folder}. Skipping.")
                    return []
                
                html_file_path = html_files[0]  # Use the first .html file found
                logger.info(f"Processing SEC document (HTML): {html_file_path}")
                doc_text = self.read_text_file(html_file_path)
                chunks = self._get_sec_chunks(doc_text, metadata)
                
                # Mark as processed
                self.mark_document_processed(metadata)
                return chunks
            else:
                # For non-SEC, use generic processing (placeholder for now)
                logger.info(f"Processing generic document: {metadata.get('raw_doc_id', 'unknown')}")
                # This would need implementation later
                return []
                
        except Exception as e:
            logger.error(f"Error processing document folder {doc_folder}: {e}", exc_info=True)
            return []

    def process_all(self) -> List[Dict]:
        """Processes all documents under base_path in parallel batches."""
        doc_folders = [
            folder for folder in self.base_path.rglob("*")
            if folder.is_dir() and (folder / "metadata.json").exists()
        ]
        
        logger.info(f"Found {len(doc_folders)} document folders")
        all_chunks = []
        processed_count = 0
        skipped_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for batch in self.batch_iterator(doc_folders, self.batch_size):
                futures = [executor.submit(self.process_document_folder, folder) for folder in batch]
                for future in futures:
                    try:
                        chunks = future.result()
                        if chunks:  # Only count if we actually got chunks
                            all_chunks.extend(chunks)
                            processed_count += 1
                        else:
                            skipped_count += 1
                    except Exception as e:
                        logger.error(f"Error processing a document folder: {e}", exc_info=True)
                        skipped_count += 1
        
        logger.info(f"Processing complete: {processed_count} documents processed, "
                   f"{skipped_count} skipped, {len(all_chunks)} total chunks created")
        return all_chunks

    def batch_iterator(self, iterable, batch_size):
        """Create batches from an iterable"""
        iterator = iter(iterable)
        for first in iterator:
            yield [first] + list(islice(iterator, batch_size - 1))

    def save_chunks(self, chunks: List[Dict], output_path: str = None):
        """Saves chunk data to individual JSON files, one per document."""
        output_dir = Path(output_path or self.output_path or "./chunks").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks_by_doc = {}
        for chunk in chunks:
            doc_id = chunk["metadata"]["raw_doc_id"]
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk)

        for doc_id, doc_chunks in chunks_by_doc.items():
            # Sanitize doc_id for use as a filename
            safe_doc_id = "".join(c for c in doc_id if c.isalnum() or c in ('-', '_')).rstrip()
            output_file = output_dir / f"{safe_doc_id}_chunks.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(doc_chunks, f, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks across {len(chunks_by_doc)} documents to {output_dir}")

    def process_and_save(self, output_path: str = None):
        """Main method to process all documents and save chunks"""
        if output_path:
            self.output_path = Path(output_path)
            
        logger.info("Starting document processing...")
        chunks = self.process_all()
        
        if chunks:
            self.save_chunks(chunks, output_path)
        else:
            logger.info("No chunks to save")

    # --- SEC Document Processing Methods (from previous implementation) ---
    
    def _process_sec_document(self, 
                             text_content: str,
                             company_id: str,
                             filing_type: str,
                             filing_date: str,
                             metadata: Dict) -> List['TextChunk']:
        """Process SEC document with HTML-aware chunking"""
        
        logger.debug(f"Processing SEC {filing_type} for {company_id}")
        
        # Try to parse as HTML first, fall back to plain text
        if self._is_html_content(text_content):
            chunks = self._process_html_content(
                text_content, company_id, filing_type, filing_date, metadata
            )
        else:
            chunks = self._process_plain_text(
                text_content, company_id, filing_type, filing_date, metadata
            )
        
        logger.debug(f"Created {len(chunks)} semantic chunks")
        return chunks

    def _is_html_content(self, text: str) -> bool:
        """Check if content appears to be HTML"""
        return bool(re.search(r'<[^>]+>', text[:1000]))

    def _process_html_content(self, 
                             html_content: str,
                             company_id: str,
                             filing_type: str,
                             filing_date: str,
                             metadata: Dict) -> List['TextChunk']:
        """Process HTML SEC document with structure preservation"""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()
            
            chunks = []
            current_section = self._detect_initial_section(soup)
            chunk_index = 0
            
            # Process different HTML elements semantically
            for element in soup.find_all(['div', 'p', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                
                # Update section context if we hit a header
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    current_section = self._update_section_from_header(element.get_text().strip())
                
                # Extract and clean text
                text = self._extract_clean_text(element)
                if not text or len(text.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Handle tables specially
                if element.name == 'table':
                    table_chunks = self._process_table_element(
                        element, company_id, filing_type, filing_date, 
                        current_section, chunk_index, metadata
                    )
                    chunks.extend(table_chunks)
                    chunk_index += len(table_chunks)
                else:
                    # Process as regular text with sentence-aware chunking
                    text_chunks = self._create_semantic_chunks(
                        text, company_id, filing_type, filing_date,
                        current_section, chunk_index, metadata, element.name
                    )
                    chunks.extend(text_chunks)
                    chunk_index += len(text_chunks)
            
            return chunks
            
        except Exception as e:
            logger.warning(f"HTML parsing failed, falling back to plain text: {e}")
            return self._process_plain_text(html_content, company_id, filing_type, filing_date, metadata)

    def _process_table_element(self, table_element, company_id: str, filing_type: str, 
                              filing_date: str, section: str, chunk_index: int, metadata: Dict) -> List['TextChunk']:
        """Process HTML tables as structured data"""
        
        # Extract table with structure
        rows = []
        for row in table_element.find_all('tr'):
            cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
            if any(cells):  # Only keep non-empty rows
                rows.append(' | '.join(cells))
        
        if not rows:
            return []
        
        table_text = '\n'.join(rows)
        
        # Detect if this looks like a financial table
        is_financial_table = self._is_financial_table(table_text)
        
        chunk = TextChunk(
            id=self._generate_chunk_id(company_id, f"{filing_type}_{filing_date}", chunk_index),
            text=table_text,
            company_id=company_id,
            source_document=f"{filing_type}_{filing_date}.html",
            chunk_index=chunk_index,
            page_number=None,
            section=section,
            doc_type=filing_type,
            period=filing_date,
            metadata={
                "element_type": "Table",
                "is_financial_table": is_financial_table,
                "row_count": len(rows),
                "contains_financial_data": is_financial_table,
                "financial_term_density": self._calculate_financial_density(table_text),
                "dominant_financial_theme": self._get_dominant_theme(table_text),
                "contains_numbers": bool(re.search(r'[\$\d,]+', table_text)),
                "word_count": len(table_text.split()),
                "char_count": len(table_text),
                "processed_at": datetime.now().isoformat(),
            }
        )
        
        return [chunk]

    def _process_plain_text(self, text_content: str, company_id: str, filing_type: str,
                           filing_date: str, metadata: Dict) -> List['TextChunk']:
        """Process plain text SEC document"""
        
        return self._create_semantic_chunks(
            text_content, company_id, filing_type, filing_date,
            "Document", 0, metadata, "PlainText"
        )

    def _create_semantic_chunks(self, text: str, company_id: str, filing_type: str,
                               filing_date: str, section: str, start_index: int,
                               metadata: Dict, element_type: str) -> List['TextChunk']:
        """Create overlapping chunks with rich metadata using sentence boundaries"""
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_sentences = []
        chunk_index = start_index
        
        for sentence in sentences:
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
                        filing_type=filing_type,
                        filing_date=filing_date,
                        section=section,
                        chunk_index=chunk_index,
                        element_type=element_type,
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) > 1 else []
                current_chunk = " ".join(overlap_sentences + [sentence])
                current_sentences = overlap_sentences + [sentence]
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = self._create_chunk_object(
                text=current_chunk.strip(),
                company_id=company_id,
                filing_type=filing_type,
                filing_date=filing_date,
                section=section,
                chunk_index=chunk_index,
                element_type=element_type,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks

    def _create_chunk_object(self, text: str, company_id: str, filing_type: str,
                            filing_date: str, section: str, chunk_index: int,
                            element_type: str, metadata: Dict) -> 'TextChunk':
        """Create a TextChunk object with comprehensive SEC metadata"""
        
        chunk_id = self._generate_chunk_id(company_id, f"{filing_type}_{filing_date}", chunk_index)
        
        # Analyze financial content
        financial_analysis = self._analyze_financial_content(text)
        
        # Create enhanced metadata
        enhanced_metadata = {
            "company_id": company_id,
            "source_document": f"{filing_type}_{filing_date}.html",
            "doc_type": filing_type,
            "period": filing_date,
            "section": section,
            "chunk_index": chunk_index,
            "element_type": element_type,
            "word_count": len(text.split()),
            "char_count": len(text),
            
            # Financial content analysis
            "contains_financial_data": financial_analysis["has_financial_data"],
            "financial_term_density": financial_analysis["term_density"],
            "dominant_financial_theme": financial_analysis["dominant_theme"],
            "contains_numbers": bool(re.search(r'[\$\d,]+', text)),
            
            # SEC specific
            "filing_type": filing_type,
            "ticker": company_id,
            "filing_date": filing_date,
            
            # Processing metadata
            "processed_at": datetime.now().isoformat(),
            "parsing_method": "semantic_html" if element_type != "PlainText" else "plain_text"
        }
        
        return TextChunk(
            id=chunk_id,
            text=text,
            company_id=company_id,
            source_document=f"{filing_type}_{filing_date}.html",
            chunk_index=chunk_index,
            page_number=None,
            section=section,
            doc_type=filing_type,
            period=filing_date,
            metadata=enhanced_metadata
        )

    # Utility methods
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving financial formatting"""
        
        # Pre-process to protect financial abbreviations and data
        text = re.sub(r'\b(Mr|Mrs|Dr|Inc|Corp|Ltd|LLC|vs|etc)\.', r'\1<DOT>', text)
        text = re.sub(r'\$(\d+[\d,]*)\.(\d+)', r'$\1<DOT>\2', text)  # Protect dollar amounts
        text = re.sub(r'(\d+)\.(\d+)%', r'\1<DOT>\2%', text)  # Protect percentages
        text = re.sub(r'Item\s+(\d+)\.', r'Item \1<DOT>', text)  # Protect SEC item numbers
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _extract_clean_text(self, element) -> str:
        """Extract clean text from HTML element"""
        if not element:
            return ""
        
        text = element.get_text()
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _detect_initial_section(self, soup) -> str:
        """Detect the initial section of the document"""
        # Look for common SEC section headers
        for header in soup.find_all(['h1', 'h2', 'h3'], limit=5):
            header_text = header.get_text().lower()
            if any(keyword in header_text for keyword in ['part i', 'part ii', 'business', 'risk factors']):
                return self._classify_section(header_text)
        return "Document"

    def _update_section_from_header(self, header_text: str) -> str:
        """Update section based on header content"""
        return self._classify_section(header_text.lower())

    def _classify_section(self, text: str) -> str:
        """Classify text into SEC document sections"""
        text_lower = text.lower()
        
        section_patterns = {
            "Business": ["business", "part i", "item 1"],
            "Risk Factors": ["risk factors", "item 1a"],
            "Legal Proceedings": ["legal proceedings", "item 3"],
            "MD&A": ["management", "discussion", "analysis", "item 7", "md&a"],
            "Financial Statements": ["financial statements", "item 8", "consolidated"],
            "Controls": ["controls", "procedures", "item 9"],
            "Directors": ["directors", "officers", "item 10"],
            "Compensation": ["compensation", "item 11"],
            "Ownership": ["ownership", "item 12"],
            "Exhibits": ["exhibits", "item 15"]
        }
        
        for section_name, keywords in section_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return section_name
        
        return "General"

    def _is_financial_table(self, table_text: str) -> bool:
        """Determine if a table contains financial data"""
        financial_indicators = [
            r'\$[\d,]+',  # Dollar amounts
            r'\d+%',      # Percentages
            r'\b(revenue|income|assets|liabilities|cash)\b',
            r'\b(million|billion|thousand)\b',
            r'\b(fiscal|quarter|year)\b'
        ]
        
        return sum(bool(re.search(pattern, table_text, re.IGNORECASE)) 
                  for pattern in financial_indicators) >= 2

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
            term_density > 0.03 or  # >3% financial terms
            bool(re.search(r'[\$\d,]+', text)) or  # Contains dollar amounts or numbers
            bool(re.search(r'\d+%', text)) or      # Contains percentages
            bool(re.search(r'\d+\.?\d*\s*(million|billion|thousand)', text, re.IGNORECASE))
        )
        
        return {
            "has_financial_data": has_financial_data,
            "term_density": round(term_density, 3),
            "dominant_theme": dominant_theme,
            "term_counts": term_counts
        }

    def _calculate_financial_density(self, text: str) -> float:
        """Calculate financial term density"""
        return self._analyze_financial_content(text)["term_density"]

    def _get_dominant_theme(self, text: str) -> str:
        """Get dominant financial theme"""
        return self._analyze_financial_content(text)["dominant_theme"]

    def _generate_chunk_id(self, company_id: str, doc_identifier: str, chunk_index: int) -> str:
        """Generate unique, deterministic chunk ID"""
        return f"{company_id}_{doc_identifier}_chunk_{chunk_index:03d}"

    @staticmethod
    def sha256_hash(text: str) -> str:
        """Generate SHA256 hash of text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


# TextChunk class definition
class TextChunk:
    """Represents a text chunk with metadata"""
    
    def __init__(self, id: str, text: str, company_id: str, source_document: str,
                 chunk_index: int, page_number: Optional[int], section: str,
                 doc_type: str, period: str, metadata: Dict):
        self.id = id
        self.text = text
        self.company_id = company_id
        self.source_document = source_document
        self.chunk_index = chunk_index
        self.page_number = page_number
        self.section = section
        self.doc_type = doc_type
        self.period = period
        self.metadata = metadata


# # Usage Example
# if __name__ == "__main__":
#     # Initialize processor
#     processor = DocumentProcessor(
#         base_path="/path/to/raw/documents",
#         output_path="/path/to/chunks",
#         chunk_size=1000,
#         max_workers=4,
#         batch_size=10
#     )
    
#     # Process all documents and save chunks
#     processor.process_and_save()