"""PDF -> ChromaDB ingestion pipeline for the policy document.

Usage:
    python ingest.py <path-to-pdf> [optional-smoke-test-query]

Idempotent: the Chroma collection is named after the PDF's SHA-256, so
re-running on the same file is a no-op. Switching PDFs creates a fresh
collection automatically.
"""

import hashlib
import sys
from pathlib import Path

import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent / "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

# Singleton chroma client + embedder shared across threads (avoids race during PersistentClient init)
_chroma_client = None
_embedder = None
_lock = __import__("threading").Lock()


def _get_chroma():
    global _chroma_client
    with _lock:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def _get_embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def pdf_fingerprint(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def collection_name_for(pdf_path: Path) -> str:
    return f"policy_{pdf_fingerprint(pdf_path)}"


def parse_pdf_to_pages(pdf_path: Path) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for p in pages:
        for j, ch in enumerate(splitter.split_text(p["text"])):
            chunks.append({
                "id": f"p{p['page']:03d}_c{j:02d}",
                "text": ch,
                "page": p["page"],
            })
    return chunks


def get_or_create_collection(pdf_path: Path):
    """Return (collection, already_existed)."""
    client = _get_chroma()
    name = collection_name_for(pdf_path)
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        return client.get_collection(name), True
    return client.create_collection(name), False


def ingest(pdf_path: Path) -> int:
    coll, already = get_or_create_collection(pdf_path)
    if already and coll.count() > 0:
        print(f"[skip] collection '{coll.name}' already populated ({coll.count()} chunks)")
        return coll.count()

    print(f"[1/4] parsing {pdf_path.name} ...")
    pages = parse_pdf_to_pages(pdf_path)
    print(f"      -> {len(pages)} non-empty pages")

    print(f"[2/4] chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = chunk_pages(pages)
    print(f"      -> {len(chunks)} chunks")

    print(f"[3/4] embedding with {EMBED_MODEL} (first run downloads ~80MB) ...")
    embedder = _get_embedder()
    vectors = embedder.encode(
        [c["text"] for c in chunks],
        show_progress_bar=False,
        batch_size=32,
    ).tolist()

    print(f"[4/4] writing to chroma collection '{coll.name}' ...")
    coll.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    print(f"done. {coll.count()} chunks indexed.")
    return coll.count()


def search(pdf_path: Path, query: str, k: int = 5) -> list[dict]:
    coll, _ = get_or_create_collection(pdf_path)
    embedder = _get_embedder()
    qvec = embedder.encode([query]).tolist()[0]
    res = coll.query(query_embeddings=[qvec], n_results=k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        hits.append({"text": doc, "page": meta["page"], "distance": dist})
    return hits


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ingest.py <path-to-pdf> [smoke-test-query]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"error: file not found: {pdf}")
        sys.exit(1)

    ingest(pdf)

    if len(sys.argv) > 2:
        q = " ".join(sys.argv[2:])
        print(f"\n--- smoke-test query: {q!r} ---")
        for i, hit in enumerate(search(pdf, q, k=3), 1):
            preview = hit["text"][:180].replace("\n", " ")
            print(f"\n[hit {i}] page {hit['page']}  distance={hit['distance']:.3f}")
            print(f"  {preview}...")
