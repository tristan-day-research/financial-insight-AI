"""
agentic_chunker.py

Local-first agentic chunking with special handling for tables.
Designed for SEC and other structured/unstructured documents without requiring prior knowledge of their format.

Features:
- Ingests PDF, HTML, TXT, CSV
- Extracts text and tables
- Agentic chunking based on semantic boundaries and headings
- Outputs chunks with type (text, table, table_row) and metadata

This module focuses only on chunking and table parsing — it does NOT handle embeddings or vector DB storage.
Call from a notebook or other pipeline.
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional

import pdfplumber
from bs4 import BeautifulSoup
import pandas as pd

# Optional LLM-based chunking
try:
    from llama_cpp import Llama
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


def extract_from_pdf(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    text_blocks, tables = [], []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            if txt.strip():
                text_blocks.append(f"\n\n[PAGE {i+1}]\n" + txt)
            try:
                page_tables = page.extract_tables()
                for t in page_tables:
                    if not t or len(t) == 0:
                        continue
                    df = pd.DataFrame(t[1:], columns=t[0]) if len(t) > 1 else pd.DataFrame(t)
                    tables.append({"page": i+1, "df": df})
            except Exception:
                pass
    return "\n".join(text_blocks), tables


def extract_from_html(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    html = open(path, "r", encoding="utf-8").read()
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style"]):
        s.extract()
    tables = []
    for i, table in enumerate(soup.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            rows.append(cols)
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows)
            tables.append({"table_id": i, "df": df})
    text = soup.get_text(separator="\n")
    return text, tables


def extract_from_csv(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    df = pd.read_csv(path)
    return df.to_csv(index=False), [{"df": df}]


def extract_from_txt(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    return open(path, "r", encoding="utf-8").read(), []


def normalize_text(text: str) -> str:
    text = text.replace('\r\n', '\n')
    text = re.sub('\n{3,}', '\n\n', text)
    return text.strip()


def agentic_split(text: str, llm_mode: str = "heuristic", model_path: Optional[str] = None, max_tokens_per_chunk: int = 800) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    boundaries = [0]
    heading_pattern = re.compile(r"^(ITEM\s+\d+\.|Item\s+\d+\.|[A-Z0-9 \-]{10,})")
    for i, ln in enumerate(lines):
        if heading_pattern.match(ln.strip()):
            boundaries.append(i)
    boundaries.append(len(lines))
    candidates = ["\n".join(lines[a:b]).strip() for a, b in zip(boundaries[:-1], boundaries[1:]) if "\n".join(lines[a:b]).strip()]
    if not candidates:
        candidates = [text]
    chunks = []
    for block in candidates:
        if len(block) < 2000 or llm_mode == "heuristic":
            chunks.append({"type": "text", "content": normalize_text(block), "meta": {}})
            continue
        prompt = (
            "Split this text into coherent chunks under " f"{max_tokens_per_chunk}" " tokens. Preserve headings and table rows.\n"
            "Return JSON list of {\"chunk\": string}.\n\n" + block
        )
        try:
            if LLM_AVAILABLE and llm_mode == "local" and model_path:
                llm = Llama(model_path=model_path)
                out = llm(prompt=prompt, max_tokens=1024)["choices"][0]["text"]
            elif OPENAI_AVAILABLE and llm_mode == "openai":
                out = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=1024, temperature=0)["choices"][0]["text"].strip()
            else:
                chunks.append({"type": "text", "content": normalize_text(block), "meta": {}})
                continue
            parsed = json.loads(out) if out.strip().startswith("[") else None
            if parsed:
                for item in parsed:
                    chunks.append({"type": "text", "content": normalize_text(item["chunk"]), "meta": {}})
            else:
                chunks.append({"type": "text", "content": normalize_text(block), "meta": {}})
        except Exception:
            chunks.append({"type": "text", "content": normalize_text(block), "meta": {}})
    return chunks


def table_to_text(df: pd.DataFrame) -> str:
    header = " | ".join(df.columns.astype(str))
    rows = [" | ".join([str(x) for x in r.tolist()]) for _, r in df.iterrows()]
    return "TABLE_START\nCOLUMNS: " + header + "\n" + "\n".join(rows) + "\nTABLE_END"


def create_row_chunks(df: pd.DataFrame, table_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    cols = list(df.columns.astype(str))
    return [{"type": "table_row", "content": f"ROW_INDEX: {i} | " + ", ".join(f"{c}: {row[c]}" for c in cols), "meta": {**table_meta, "row_index": int(i)}} for i, row in df.iterrows()]


def process_file_for_chunks(path: str, llm_mode: str = "heuristic", llama_model_path: Optional[str] = None) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text, tables = extract_from_pdf(path)
    elif ext in (".html", ".htm"):
        text, tables = extract_from_html(path)
    elif ext == ".csv":
        text, tables = extract_from_csv(path)
    else:
        text, tables = extract_from_txt(path)
    text = normalize_text(text)
    chunks = agentic_split(text, llm_mode=llm_mode, model_path=llama_model_path)
    for t in tables:
        df = t.get("df")
        tmeta = {k: v for k, v in t.items() if k != "df"}
        chunks.append({"type": "table", "content": table_to_text(df), "meta": tmeta})
        chunks.extend(create_row_chunks(df, tmeta))
    return chunks
