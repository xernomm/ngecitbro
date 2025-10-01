# bot.py
"""
RAG-focused helper for a chatbot (tools-free).
Provides:
- initialize(chroma_path, collection_name)
- process_file(file_path: str, doc_id: str)
- add_manual_context(doc_id: str, text: str)
- get_relevant_context(user_input: str, doc_ids: Optional[List[str]] = None, n_results: int = 5)
- generate_answer(user_input: str, doc_ids: Optional[List[str]] = None, model: str = DEFAULT_LLM_MODEL, n_results: int = 5)
- delete_doc(doc_id: str)

Notes:
- Uses Ollama for embedding/chat and ChromaDB for vector storage.
- Adjust EMBED_MODEL and DEFAULT_LLM_MODEL to match your environment.
"""

import os
import uuid
import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
import inspect

# FastAPI helpful exceptions (file is controller-agnostic but kept for compatibility)
from fastapi import HTTPException

import docx  # for .docx
import ollama
import chromadb
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_context"
EMBED_MODEL = "mxbai-embed-large:latest"  # adjust if needed
DEFAULT_LLM_MODEL = "deepseek-r1:latest"               # adjust if needed in Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Internal global handles
_client = None
_collection = None


def initialize(chroma_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
    """
    Initialize Chroma persistent client and collection. Safe to call multiple times.
    Returns the collection object.
    """
    global _client, _collection
    if _client is None:
        logger.info("Initializing Chroma PersistentClient at %s", chroma_path)
        _client = chromadb.PersistentClient(path=chroma_path)
    if _collection is None:
        _collection = _client.get_or_create_collection(name=collection_name)
    return _collection


# initialize on import
initialize()

# -----------------------
# Utility helpers
# -----------------------
def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys with None values from metadata."""
    return {k: v for k, v in metadata.items() if v is not None}


def _existing_ids_start_with(prefix: str) -> bool:
    """
    Check if any id in collection starts with prefix.
    Fallback-safe: retrieves ids and checks startswith.
    """
    try:
        all_ids = _collection.get().get("ids", [])
        return any(i.startswith(prefix) for i in all_ids)
    except Exception as e:
        logger.warning("Failed to check existing ids for prefix %s: %s", prefix, e)
        return False

def _call_chat(model, messages, **kwargs):
    """
    Call ollama_client.chat while only passing keyword args that the client's
    chat() accepts. This avoids TypeError: got an unexpected keyword argument 'temperature'.
    If the client's chat() expects 'prompt' instead of 'messages', we coerce.
    """
    func = getattr(ollama_client, "chat")
    try:
        sig = inspect.signature(func)
        allowed_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        params = {}

        # model param
        if "model" in sig.parameters:
            params["model"] = model
        else:
            # some clients accept model as first positional - will try positional fallback below
            pass

        # messages / prompt param handling
        if "messages" in sig.parameters:
            params["messages"] = messages
        elif "prompt" in sig.parameters:
            # convert list-of-messages into a single prompt string
            if isinstance(messages, list):
                params["prompt"] = "\n".join([m.get("content", "") for m in messages])
            else:
                params["prompt"] = messages
        else:
            # no messages/prompt accepted as kwarg -> will attempt positional call
            pass

        # include allowed kwargs (e.g., if this version supports temperature, max_tokens)
        params.update(allowed_kwargs)

        logger.debug("Calling ollama_client.chat with params: %s", list(params.keys()))
        return func(**params)
    except TypeError as te:
        # signature matching may still fail, fall back to positional call
        logger.warning("Chat wrapper TypeError: %s — falling back to positional call", te)
        try:
            return func(model, messages)
        except Exception as e:
            logger.error("Fallback positional call to chat() failed: %s", e)
            raise
    except Exception as e:
        logger.warning("Failed to inspect chat signature: %s — trying positional call", e)
        return func(model, messages)

# -----------------------
# Robust embedding wrapper
# -----------------------
def _get_embeddings_for_inputs(inputs):
    """
    inputs: str or list[str]
    returns: list[embedding] (one per input)
    Robustly handles several Ollama response formats.
    """
    try:
        resp = ollama_client.embed(model=EMBED_MODEL, input=inputs)
    except Exception as e:
        logger.error("Embedding call to Ollama failed: %s", e)
        raise RuntimeError(f"Embedding call failed: {e}") from e

    # resp may be dict-like or object-like
    if isinstance(resp, dict):
        if "embeddings" in resp and resp["embeddings"]:
            return resp["embeddings"]
        if "embedding" in resp and resp["embedding"]:
            emb = resp["embedding"]
            # if single vector provided as list of floats => wrap
            if isinstance(emb[0], (list, tuple)):
                return emb
            return [emb]
    # try attributes
    emb_attr = getattr(resp, "embeddings", None) or getattr(resp, "embedding", None)
    if emb_attr:
        if isinstance(emb_attr[0], (list, tuple)):
            return emb_attr
        return [emb_attr]

    # last resort: if response itself is a vector
    if isinstance(resp, (list, tuple)):
        return [list(resp)]

    raise RuntimeError("Unrecognized embedding response format from Ollama.")


# -----------------------
# Text extraction (PDF per-page, DOCX, plain)
# -----------------------
def _extract_text_pages_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from PDF.
    Returns list of {"page": int, "text": str}
    """
    logger.info("Extracting PDF per-page: %s", file_path)
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as e:
            logger.warning("Failed to extract text from page %d of %s: %s", i + 1, file_path, e)
            t = ""
        pages.append({"page": i + 1, "text": t})
    return pages


def _extract_text_from_docx(file_path: str) -> str:
    logger.info("Extracting DOCX: %s", file_path)
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()


def _extract_text_from_plain_text(file_path: str) -> str:
    logger.info("Reading plain text file: %s", file_path)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read().strip()


# -----------------------
# Indexing: accepts string OR list-of-pages
# -----------------------
def _index_text(text_or_pages, doc_id: str, source_filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    """
    text_or_pages: either a single string, or list of {"page":int,"text":str}
    Stores chunks with metadata: doc_id, source, page, chunk_index, chunk_hash
    """
    # Normalize into page-like items
    items = []
    if isinstance(text_or_pages, str):
        items = [{"page": None, "text": text_or_pages}]
    else:
        items = text_or_pages

    all_chunks = []
    all_metadatas = []
    all_ids = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for page_info in items:
        page_num = page_info.get("page")
        page_text = page_info.get("text", "") or ""
        raw_chunks = [c.strip() for c in splitter.split_text(page_text) if c.strip()]
        for idx, c in enumerate(raw_chunks):
            chunk_hash = hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{doc_id}#chunk_p{page_num or 0}_{idx}"
            meta = sanitize_metadata({
                "doc_id": doc_id,
                "source": source_filename,
                "page": page_num,
                "chunk_index": idx,
                "chunk_hash": chunk_hash
            })
            all_chunks.append(c)
            all_metadatas.append(meta)
            all_ids.append(chunk_id)

    if not all_chunks:
        logger.warning("No chunks produced for doc_id %s", doc_id)
        return {"status": "no_chunks", "message": "Tidak ada chunk setelah pemisahan teks."}

    # Create embeddings
    try:
        embeddings = _get_embeddings_for_inputs(all_chunks)
        if len(embeddings) != len(all_chunks):
            logger.warning("Embedding count mismatch; trying per-chunk fallback.")
            embeddings = [ _get_embeddings_for_inputs(c)[0] for c in all_chunks ]
    except Exception as e:
        logger.error("Failed to create embeddings: %s", e)
        raise

    # Add to Chroma
    _collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas
    )
    logger.info("Indexed %d chunks for doc_id %s", len(all_ids), doc_id)
    return {"status": "ok", "chunks_indexed": len(all_ids), "doc_id": doc_id}


# -----------------------
# Public: process_file (uses per-page extraction for PDFs)
# -----------------------
def process_file(file_path: str, doc_id: str) -> Dict[str, Any]:
    """
    Process uploaded file: detect type, extract text, and index it.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if _existing_ids_start_with(f"{doc_id}#chunk"):
        logger.info("Document '%s' already indexed. Skipping.", doc_id)
        return {"status": "exists", "message": "Dokumen sudah diindeks."}

    filename = os.path.basename(file_path)
    _, extension = os.path.splitext(filename)
    extension = extension.lower()

    if extension == ".pdf":
        pages = _extract_text_pages_from_pdf(file_path)
        return _index_text(pages, doc_id, filename)
    elif extension == ".docx":
        extracted_text = _extract_text_from_docx(file_path)
        return _index_text(extracted_text, doc_id, filename)
    elif extension in [".txt", ".md", ".py", ".json", ".csv", ".html"]:
        extracted_text = _extract_text_from_plain_text(file_path)
        return _index_text(extracted_text, doc_id, filename)
    else:
        logger.warning("Unsupported file type: %s for file %s", extension, filename)
        raise HTTPException(status_code=415, detail=f"Tipe file '{extension}' tidak didukung.")


# -----------------------
# Add manual context (textarea)
# -----------------------
def add_manual_context(doc_id: str, text: str) -> Dict[str, Any]:
    """
    Add a manual context chunk (from textarea) under doc_id.
    """
    if not text or not text.strip():
        return {"status": "empty", "message": "No text provided."}

    text = text.strip()
    embedding = _get_embeddings_for_inputs(text)[0]

    existing_ids = _collection.get().get("ids", [])
    n = len([i for i in existing_ids if i.startswith(f"{doc_id}#manual")])
    new_id = f"{doc_id}#manual{n}"

    _collection.add(
        ids=[new_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[sanitize_metadata({"doc_id": doc_id, "source": "manual"})]
    )
    logger.info("Added manual context for doc_id %s id=%s", doc_id, new_id)
    return {"status": "ok", "id": new_id, "doc_id": doc_id}


# -----------------------
# Build where clause for doc filtering
# -----------------------
def _build_where_clause_for_doc_ids(doc_ids: Optional[List[str]]):
    """
    Build a Chroma 'where' clause. If doc_ids is None -> return None (no filtering).
    """
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": doc_ids[0]}
    return {"$or": [{"doc_id": d} for d in doc_ids]}


# -----------------------
# Retrieval: returns structured hits
# -----------------------
def get_relevant_context(user_input: str, doc_ids: Optional[List[str]] = None, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-n relevant chunks for the given user_input.
    Returns a list of hit dicts with fields: id (may be None), document, metadata, distance
    """
    if not user_input or not user_input.strip():
        return []

    user_input = user_input.strip()
    user_emb = _get_embeddings_for_inputs(user_input)[0]

    where_clause = _build_where_clause_for_doc_ids(doc_ids)

    try:
        # NOTE: "ids" is not allowed in include on some Chroma versions -> don't include it
        results = _collection.query(
            query_embeddings=[user_emb],
            where=where_clause,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
    except ValueError as ve:
        # Fallback: try without include (older/newer chroma)
        logger.warning("Chroma query include failed (%s). Retrying without include.", ve)
        results = _collection.query(
            query_embeddings=[user_emb],
            where=where_clause,
            n_results=n_results
        )
    except Exception as e:
        logger.exception("Chroma query failed")
        raise

    # results may or may not contain keys depending on chroma version
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    dists = results.get("distances", [[]])[0] if results.get("distances") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []

    hits = []
    for i in range(len(docs)):
        hit_id = ids[i] if i < len(ids) else None
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        hits.append({
            "id": hit_id,
            "document": docs[i],
            "metadata": meta,
            "distance": dist
        })
    return hits



# -----------------------
# Reranker using Ollama
# -----------------------
RERANKER_SYSTEM = "You are a strict reranker. Output ONLY a JSON array (e.g. [\"id1\",\"id2\"]) sorted from most to least relevant."
def rerank_hits(question: str, hits: List[Dict[str, Any]], top_k: int = 5, model: str = DEFAULT_LLM_MODEL) -> List[Dict[str, Any]]:
    if not hits:
        return []

    candidate_lines = []
    for h in hits:
        snippet = h["document"][:800].replace("\n", " ").strip()
        meta = h.get("metadata", {})
        candidate_lines.append(f"ID: {h['id']}\nMETA: {json.dumps(meta)}\nTEXT: {snippet}")

    candidates_block = "\n\n---\n\n".join(candidate_lines)
    user_prompt = f"Question: {question}\n\nCandidates:\n\n{candidates_block}\n\nReturn a JSON array of candidate IDs sorted from most to least relevant to the question."

    try:
        resp = _call_chat(
            model="llama3",
            messages=[{"role": "system", "content": RERANKER_SYSTEM},
                      {"role": "user", "content": user_prompt}],
            temperature=0  # will be filtered out if client doesn't accept it
        )

        # robust extraction of text
        text = ""
        if isinstance(resp, dict):
            if "message" in resp:
                msg = resp["message"]
                text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            elif "choices" in resp and resp["choices"]:
                choice = resp["choices"][0]
                text = (choice.get("message", {}) or {}).get("content", "") or choice.get("text", "") or ""
        else:
            msg = getattr(resp, "message", None)
            if msg:
                text = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            else:
                choices = getattr(resp, "choices", None)
                if choices:
                    first = choices[0]
                    text = getattr(first, "text", "") or (first.get("message", {}) or {}).get("content", "")

        text = (text or "").strip()
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            ordered_ids = json.loads(json_str)
            id_to_hit = {h['id']: h for h in hits}
            ordered_hits = [id_to_hit[i] for i in ordered_ids if i in id_to_hit]
            return ordered_hits[:top_k]
    except Exception as e:
        logger.warning("Reranker failed or unparsable: %s", e)

    with_distance = [h for h in hits if h.get("distance") is not None]
    if with_distance:
        ordered = sorted(with_distance, key=lambda x: x["distance"])
        return ordered[:top_k]
    return hits[:top_k]


# -----------------------
# Assemble labeled context string
# -----------------------
def assemble_context_string(hits: List[Dict[str, Any]], max_chars: int = 5000, max_snippet_chars: int = 3000) -> str:
    """
    Build a single string with labelled chunks, limited by max_chars (approx).
    Each chunk includes its ID and a truncated snippet of max_snippet_chars.
    """
    parts = []
    total = 0
    for h in hits:
        snippet = h["document"].strip().replace("\n", " ")
        # truncate each chunk to avoid huge context
        if len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars].rsplit(" ", 1)[0] + "..."
        meta = h.get("metadata", {})
        src = meta.get("source")
        page = meta.get("page")
        label = f"[{h.get('id') or 'unknown'}] source={src}{(' page='+str(page)) if page else ''}"
        part = f"{label}\n{snippet}\n"
        if total + len(part) > max_chars and parts:
            break
        parts.append(part)
        total += len(part)
    return "\n---\n".join(parts)



# -----------------------
# Strict system instruction for answer generation
# -----------------------
SYSTEM_INSTRUCTIONS_STRICT = """
Anda adalah asisten yang hanya boleh menggunakan informasi yang diberikan di bagian 'CONTEXT' untuk menjawab pertanyaan.
- Jangan menambahkan atau mengarang informasi di luar konteks.
- Untuk setiap fakta penting yang Anda sebutkan, sertakan citation singkat di akhir klausa dalam bentuk [CHUNK_ID].
- Jika jawaban tidak ada di konteks, jawab: "Maaf, saya tidak menemukan informasi itu dalam dokumen yang diberikan."
- Jawaban harus singkat, jelas, dan sertakan daftar sumber (chunk ids) yang mendukung jawaban.
"""


# -----------------------
# Main generate_answer (retrieve -> rerank -> assemble -> chat)
# -----------------------
# --- use _call_chat in generate_answer ---
def generate_answer(user_input: str,
                    doc_ids: Optional[List[str]] = None,
                    model: str = DEFAULT_LLM_MODEL,
                    n_results: int = 8,
                    system_instructions: Optional[str] = None) -> Dict[str, Any]:
    if not user_input or not user_input.strip():
        return {"answer": "", "context_hits": [], "doc_ids_used": doc_ids, "model": model}

    hits = get_relevant_context(user_input, doc_ids=doc_ids, n_results=n_results)
    if not hits:
        return {"answer": "Maaf, tidak ada konteks relevan yang ditemukan.", "context_hits": [], "model": model}

    top_hits = rerank_hits(user_input, hits, top_k=5, model=model)

    # Note: tune these limits to fit your model's context window
    context_str = assemble_context_string(top_hits, max_chars=5000, max_snippet_chars=3000)
    if context_str: 
        print(context_str)
    sys_instr = system_instructions or SYSTEM_INSTRUCTIONS_STRICT
    messages = [
        {"role": "system", "content": sys_instr},
        {"role": "system", "content": f"KONTEXT (gunakan hanya jika relevan):\n{context_str}"},
        {"role": "user", "content": user_input}
    ]

    logger.info("Calling LLM model %s with %d top chunks (context ~%d chars)", model, len(top_hits), len(context_str))
    try:
        resp = _call_chat(
            model=model,
            messages=messages,
            temperature=0,   # safe: will be ignored if unsupported
            # max_tokens=800   # safe: will be ignored if unsupported
        )
    except Exception as e:
        logger.error("LLM chat call failed: %s", e)
        return {"answer": "Terjadi kesalahan saat memanggil model.", "context_hits": top_hits, "model": model}

    # robust extraction (same as before)...
    content = ""
    if isinstance(resp, dict):
        if "message" in resp:
            msg = resp["message"]
            if isinstance(msg, dict):
                content = msg.get("content", "") or ""
            else:
                content = str(msg) or ""
        elif "choices" in resp and resp["choices"]:
            choice = resp["choices"][0]
            if isinstance(choice, dict):
                content = (choice.get("message", {}) or {}).get("content", "") or choice.get("text", "") or ""
            else:
                content = getattr(choice, "text", "") or str(choice)
    else:
        msg = getattr(resp, "message", None)
        if msg is not None:
            if isinstance(msg, dict):
                content = msg.get("content", "") or ""
            else:
                content = getattr(msg, "content", "") or ""
        else:
            choices = getattr(resp, "choices", None)
            if choices:
                try:
                    first = choices[0]
                    if isinstance(first, dict):
                        m = first.get("message", {})
                        content = m.get("content", "") or first.get("text", "") or ""
                    else:
                        content = getattr(first, "text", "") or ""
                except Exception:
                    content = ""

    content = (content or "").strip()

    return {
        "answer": content,
        "context": context_str,
        "context_hits": top_hits,
        "doc_ids_used": doc_ids,
        "model": model
    }

# -----------------------
# Delete document (all chunks starting with doc_id#)
# -----------------------
def delete_doc(doc_id: str) -> Dict[str, Any]:
    """
    Remove all items whose ids start with doc_id# from the collection.
    Useful for replacing dynamic docs.
    """
    all_ids = _collection.get().get("ids", [])
    to_delete = [i for i in all_ids if i.startswith(f"{doc_id}#")]
    if not to_delete:
        return {"status": "not_found", "deleted": 0}
    _collection.delete(ids=to_delete)
    logger.info("Deleted %d items for doc_id %s", len(to_delete), doc_id)
    return {"status": "ok", "deleted": len(to_delete)}


# === Functions intended to be called by controller (app.py) ===
# - process_file(file_path: str, doc_id: str)
# - add_manual_context(doc_id: str, text: str)
# - get_relevant_context(user_input: str, doc_ids: Optional[List[str]] = None, n_results: int = 5)
# - generate_answer(user_input: str, doc_ids: Optional[List[str]] = None, model: str = DEFAULT_LLM_MODEL, n_results: int = 5)
# - delete_doc(doc_id: str)
#
# Example usage (pseudocode):
#   process_file('/tmp/upload.pdf', 'doc-2025-01')
#   add_manual_context('doc-2025-01', 'Catatan: versi final')
#   resp = generate_answer('Apa poin utama pasal 2?', doc_ids=['doc-2025-01'])
#   print(resp['answer'])
