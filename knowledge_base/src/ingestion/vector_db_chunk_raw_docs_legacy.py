import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from sec_parser import Edgar10QParser, ParsingOptions

import re
from datetime import datetime

import sec_parser as sp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge_base.config.settings import get_settings

logger = logging.getLogger(__name__)

# --- CHANGE: Renamed class for broader purpose ---
class DocumentProcessor:
    """
    Processes various documents by routing them to the correct parser and chunker
    based on their metadata. Handles both SEC filings and generic text documents.
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_path = Path(self.settings.data.raw_docs_base_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.data.chunk_tokens,
            chunk_overlap=self.settings.data.chunk_overlap,
            length_function=len
    )
        # Initialize the SEC parser
        self.sec_parser = Edgar10QParser(ParsingOptions())

    def sha256_hash(self, text: str) -> str:
        """Generate a SHA-256 hash for deduplication and tracking."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def read_text_file(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # --- CHANGE: Simplified to be more generic ---
    def load_metadata(self, doc_folder: Path) -> Dict:
        """Loads metadata.json. Expects at least a 'file_path' and 'raw_doc_id'."""
        metadata_file = doc_folder / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"No metadata.json in {doc_folder}")

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "file_path" not in metadata or "raw_doc_id" not in metadata:
            raise ValueError(f"metadata.json in {doc_folder} must contain 'file_path' and 'raw_doc_id'")
        return metadata

    def get_text_file_path(self, metadata: Dict, doc_folder: Path) -> Path:
        file_path = Path(metadata["file_path"]).expanduser()
        if file_path.is_absolute():
            return file_path
        # Assumes project root is 3 levels up from this file's location
        project_root = Path(__file__).resolve().parents[3]
        resolved_path = (project_root / file_path).resolve()
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Document file not found at {resolved_path}")
        return resolved_path

    # --- For SEC-specific chunking ---
    # def _get_sec_chunks(self, doc_text: str, metadata: Dict) -> List[Dict]:
    #     """Creates chunks for SEC documents using sec-parser for high-quality splitting."""
    #     # elements = self.sec_splitter.parse(doc_text)
    #     elements = sp.parse_filing(doc_text)
    #     chunks = [e.text for e in elements]
        
    #     chunk_dicts = []
    #     for idx, chunk in enumerate(chunks):
    #         chunk_metadata = {
    #             "chunk_id": self.sha256_hash(chunk),
    #             "raw_doc_id": metadata["raw_doc_id"], 
    #             "chunk_index": idx,
    #             "source": metadata.get("source", "SEC EDGAR"),
    #             "doc_type": "sec_filing",
    #             # SEC specific fields
    #             "filing_type": metadata.get("type"),
    #             "ticker": metadata.get("ticker"),
    #             "filing_date": metadata.get("date"),
    #             "accession_number": metadata.get("accession_number"),
    #             "path": metadata.get("file_path"),
    #             "vector_store_id": None,
    #         }
    #         chunk_dicts.append({"text": chunk, "metadata": chunk_metadata})
    #     return chunk_dicts


    def _get_sec_chunks(self, doc_text: str, metadata: Dict) -> List[Dict]:
        """Creates structured chunks for SEC documents with semantic metadata."""
        # Parse with semantic understanding
        try:
            # For modern sec-parser versions (v0.56+)
            parser = sp.Edgar10QParser() if metadata.get("type") == "10-Q" else sp.Edgar10KParser()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Invalid section type for")
                elements = parser.parse(doc_text)
            tree = sp.TreeBuilder().build(elements)
            nodes = list(tree)
        except Exception as e:
            print(f"Failed semantic parsing: {e}")
            # Fallback to simple HTML parsing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(doc_text, 'html.parser')
            nodes = [sp.TextElement(tag) for tag in soup.find_all(string=True) if tag.strip()]

        chunk_dicts = []
        current_section = "Document"
        
        for idx, node in enumerate(nodes):
            # Clean and prepare text
            text = re.sub(r'\s+', ' ', node.text).strip()
            if not text or len(text) < 25:  # Skip very small chunks
                continue
                
            # Track section hierarchy
            if isinstance(node, (sp.TitleElement, sp.TopSectionTitle)):
                current_section = text
                continue
                
            # Prepare enhanced metadata
            chunk_metadata = {
                "chunk_id": self.sha256_hash(text),
                "raw_doc_id": metadata["raw_doc_id"],
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
                # Semantic metadata
                "section": current_section,
                "element_type": node.__class__.__name__,
                "parsing_method": "semantic" if 'tree' in locals() else "html_fallback",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Handle tables differently
            if isinstance(node, sp.TableElement):
                chunk_metadata["table_summary"] = node.summary
                text = f"TABLE: {node.summary}\n{text[:500]}..."  # Truncate large tables
                
            chunk_dicts.append({
                "text": f"{current_section}: {text}" if current_section != "Document" else text,
                "metadata": chunk_metadata
            })

        return chunk_dicts


    # --- CHANGE: New method for generic document chunking ---
    def _get_generic_chunks(self, doc_text: str, metadata: Dict) -> List[Dict]:
        """Creates chunks for generic documents using a standard text splitter."""
        chunks = self.text_splitter.split_text(doc_text)
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

    # --- CHANGE: Main processing logic now acts as a dispatcher ---
    def process_document_folder(self, doc_folder: Path) -> List[Dict]:
        """
        Processes a single document folder by dispatching to the correct
        chunking method based on the document's metadata.
        """
        metadata = self.load_metadata(doc_folder)

        # SEC document logic: only process .html in the folder
        if "accession_number" in metadata:
            html_files = list(doc_folder.glob("*.html"))
            if not html_files:
                logger.warning(f"No .html file found in SEC doc folder: {doc_folder}. Skipping.")
                return []
            html_file_path = html_files[0]  # Use the first .html file found
            logger.info(f"Processing SEC document (HTML): {html_file_path}")
            doc_text = self.read_text_file(html_file_path)
            return self._get_sec_chunks(doc_text, metadata)
        else:
            # For non-SEC, use the file_path in metadata as before
            text_file_path = self.get_text_file_path(metadata, doc_folder)
            doc_text = self.read_text_file(text_file_path)
            logger.info(f"Processing generic document: {metadata['raw_doc_id']}")
            return self._get_generic_chunks(doc_text, metadata)

    def process_all(self) -> List[Dict]:
        """Processes all documents under base_path in parallel batches."""
        doc_folders = [
            folder for folder in self.base_path.rglob("*")
            if folder.is_dir() and (folder / "metadata.json").exists()
        ]
        all_chunks = []
        with ThreadPoolExecutor(max_workers=self.settings.data.local_concurrency) as executor:
            for batch in self.batch_iterator(doc_folders, self.settings.data.batch_size_local):
                futures = [executor.submit(self.process_document_folder, folder) for folder in batch]
                for future in futures:
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Error processing a document folder: {e}", exc_info=True)
        return all_chunks

    def batch_iterator(self, iterable, batch_size):
        iterator = iter(iterable)
        for first in iterator:
            yield [first] + list(islice(iterator, batch_size - 1))

    # --- CHANGE: Generalized to save files by the universal 'raw_doc_id' ---
    def save_chunks(self, chunks: List[Dict], output_path: str = None):
        """Saves chunk data to individual JSON files, one per document."""
        output_dir = Path(output_path or self.settings.data.processed_text_chunk_path).resolve()
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