import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge_base.config.settings import get_settings

logger = logging.getLogger(__name__)

class Docue:
    """
    Processes SEC filing documents by:
    1. Loading metadata for each document
    2. Reading the associated raw text file
    3. Splitting the text into chunks
    4. Attaching metadata to each chunk for downstream processing (e.g., embeddings, RAG)
    5. Optionally saving chunks to disk
    """

    def __init__(self):
        self.settings = get_settings()

        # Base path where all raw SEC filings are stored
        self.base_path = Path(self.settings.data.raw_docs_base_path)

        # Initialize the LangChain text splitter for chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.data.chunk_tokens,     # characters per chunk (configurable)
            chunk_overlap=self.settings.data.chunk_overlap, # overlap between chunks for context retention
            length_function=len                             # currently counts characters; can be swapped for tokens
        )

    def sha256_hash(self, text: str) -> str:
        """Generate a SHA-256 hash for deduplication and tracking."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def read_text_file(self, path: Path) -> str:
        """Reads the entire content of a text file."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_metadata(self, doc_folder: Path) -> Dict:
        """
        Loads and validates metadata.json.
        Requires:
        - file_path: Absolute/relative path to the raw document.
        - accession_number: Unique SEC document ID.
        - type: Filing type (e.g., 10-Q, 10-K).
        - date: Filing date.
        """
        metadata_file = doc_folder / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"No metadata.json in {doc_folder}")

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Validate required fields
        required_fields = ["file_path", "accession_number", "type", "date"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required field '{field}' in metadata.json at {doc_folder}")

        return metadata

    # def get_text_file_path(self, metadata: Dict, doc_folder: Path) -> Path:
    #     """Returns the resolved path to the raw text file."""
    #     return Path(metadata["file_path"])

    def get_text_file_path(self, metadata: Dict, doc_folder: Path) -> Path:
        """
        Resolves the text file path from metadata.
        Handles:
        - Absolute paths
        - Project-relative paths (starting with 'knowledge_base/')
        - Paths relative to the document folder
        """
        file_path = Path(metadata["file_path"]).expanduser()

        # Case 1: Absolute path → return as-is
        if file_path.is_absolute():
            return file_path

        # Case 2: Project-relative ("knowledge_base/...") → resolve from project root
        if str(file_path).startswith("knowledge_base/"):
            # CORRECTED: Go up one more level to the actual project root directory
            # that contains the 'knowledge_base' folder.
            project_root = Path(__file__).resolve().parents[3]
            resolved_path = (project_root / file_path).resolve()
        else:
            # Case 3: Path relative to the document folder
            resolved_path = (doc_folder / file_path).resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Document file not found at {resolved_path}. "
                f"Metadata points to: {metadata['file_path']}"
            )

        return resolved_path

    def chunk_document(self, doc_text: str, metadata: Dict) -> List[Dict]:
        """
        Splits the raw text into smaller chunks and attaches relevant metadata
        so each chunk can be linked back to the original filing.
        """
        chunks = self.splitter.split_text(doc_text)
        chunk_dicts = []

        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_id": self.sha256_hash(chunk),
                "raw_doc_id": metadata.get("accession_number") or metadata.get("raw_doc_id"),
                "chunk_index": idx,
                "chunk_id": self.sha256_hash(chunk),
                "source": metadata.get("source", "SEC EDGAR"),
                "filing_type": metadata.get("type") or metadata.get("filing_type"),
                "ticker": metadata.get("ticker"),
                "filing_date": metadata.get("date") or metadata.get("filing_date"),
                "accession_number": metadata.get("accession_number"),
                "path": metadata.get("file_path") or metadata.get("raw_file", {}).get("path"),
                "vector_store_id": None,
            }
            chunk_dicts.append({"text": chunk, "metadata": chunk_metadata})

        return chunk_dicts

    def batch_iterator(self, iterable, batch_size):
        """
        Yield successive fixed-size batches from an iterable.
        Used to limit memory load and control throughput.
        """
        iterator = iter(iterable)
        for first in iterator:
            yield [first] + list(islice(iterator, batch_size - 1))

    def process_document_folder(self, doc_folder: Path) -> List[Dict]:
        """Processes a single document folder into chunks."""
        metadata = self.load_metadata(doc_folder)
        text_file = self.get_text_file_path(metadata, doc_folder)
        doc_text = self.read_text_file(text_file)
        return self.chunk_document(doc_text, metadata)

    def process_all(self) -> List[Dict]:
        """
        Processes all SEC documents under base_path in parallel batches.
        Returns a combined list of all chunks.
        """
        doc_folders = [
            folder for folder in self.base_path.rglob("*")
            if folder.is_dir() and (folder / "metadata.json").exists()
        ]
        all_chunks = []

        with ThreadPoolExecutor(max_workers=self.settings.data.local_concurrency) as executor:
            for batch in self.batch_iterator(doc_folders, self.settings.data.batch_size_local):
                futures = [executor.submit(self.process_document_folder, folder) for folder in batch]
                for future in futures:
                    chunks = future.result()
                    all_chunks.extend(chunks)

        return all_chunks

    def save_chunks(self, chunks: List[Dict], output_path: str = None):
        """
        Saves chunk data to individual JSON files in the processed_text_chunk_path directory.
        Each file is named after the document's accession_number.
        Creates directory if it doesn't exist.
        """
        if output_path:
            output_dir = Path(output_path)
        else:
            # Use the processed_text_chunk_path from settings as is
            output_dir = Path(self.settings.data.processed_text_chunk_path).resolve()

        print("output_dir", output_dir)

        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group chunks by document ID
        chunks_by_doc = {}
        for chunk in chunks:
            doc_id = chunk["metadata"].get("accession_number")
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk)

        # Save each document's chunks to a separate file
        for doc_id, doc_chunks in chunks_by_doc.items():
            output_file = output_dir / f"{doc_id}_chunks.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(doc_chunks, f, indent=2)
        logger.info(f"Saved {len(chunks)} chunks across {len(chunks_by_doc)} documents to {output_dir}")