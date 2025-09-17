# app.py
"""
FastAPI controller for RAG-only bot (bot.py).
Place this file together with bot.py in the same folder.

Endpoints:
- POST /contexts/upload  -> form: doc_id (str), file (pdf)
- POST /contexts/manual  -> json: {"doc_id": "..."(optional), "text": "..."}
- GET  /contexts         -> list indexed doc_ids
- GET  /contexts/{doc_id}/retrieve?q=...&n_results=5
- DELETE /contexts/{doc_id}
- POST /chat             -> json: {"user_input": "...", "doc_ids": [...], "n_results": 5, "model": "...", "system_instructions": "..."}
- GET /health
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import shutil
import uuid
import logging
import random
import string

# ensure local folder on path so import bot works even when uvicorn started elsewhere
sys.path.append(os.path.dirname(__file__) or ".")

import bot  # the bot.py you already prepared

# Config
CONTEXTS_FOLDER = "./contexts"
os.makedirs(CONTEXTS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

app = FastAPI(title="RAG Chatbot Controller")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_input: str
    doc_ids: Optional[List[str]] = None
    n_results: Optional[int] = 5
    model: Optional[str] = None
    system_instructions: Optional[str] = None


class ManualContextRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None  # optional: allow caller to specify doc_id


@app.post("/contexts/upload")
async def upload_context(file: UploadFile = File(...), doc_id: Optional[str] = Form(None)):
    """
    Save uploaded file and index it in vector store.
    """
    final_doc_id = doc_id.strip() if (doc_id and doc_id.strip()) else str(uuid.uuid4())
    doc_folder = os.path.join(CONTEXTS_FOLDER, final_doc_id)
    os.makedirs(doc_folder, exist_ok=True)
    filename = file.filename or f"upload-{uuid.uuid4()}"
    dest_path = os.path.join(doc_folder, filename)

    try:
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        logger.exception("Gagal menyimpan file yang diunggah")
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {e}")

    try:
        result = bot.process_file(dest_path, final_doc_id)
        if result.get("status") == "exists":
            return {
                "status": "exists",
                "doc_id": final_doc_id,
                "message": "Dokumen sudah ada di vector store. File berhasil disimpan.",
                "saved_to": dest_path
            }
        return {
            "status": "ok",
            "doc_id": final_doc_id,
            "saved_to": dest_path,
            "index_result": result
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Gagal memproses file: {dest_path}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses file: {str(e)}")


@app.post("/contexts/manual")
def add_manual_context(req: ManualContextRequest):
    """
    Add a manual context chunk. If doc_id not provided, generate a short random one.
    """
    try:
        final_doc_id = req.doc_id.strip() if (req.doc_id and req.doc_id.strip()) else ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        res = bot.add_manual_context(final_doc_id, req.text)
    except Exception as e:
        logger.exception("add_manual_context failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "doc_id": final_doc_id, "result": res}


@app.get("/contexts")
def list_contexts():
    """
    List unique doc_id values currently present in the vector store,
    dan sertakan nama file (1 file per folder).
    
    Response:
    {
      "contexts": [
        { "doc_id": "doc-1", "filename": "dokumen.pdf" },
        { "doc_id": "abc123", "filename": null }
      ]
    }
    """
    try:
        # ambil doc_id dari vector store
        data = getattr(bot, "_collection").get()
        metadatas = data.get("metadatas", []) or []
        doc_ids = set()
        for m in metadatas:
            if isinstance(m, dict) and m.get("doc_id"):
                doc_ids.add(m.get("doc_id"))

        contexts_list = []
        for doc_id in sorted(list(doc_ids)):
            folder = os.path.join(CONTEXTS_FOLDER, doc_id)
            filename = None
            try:
                if os.path.isdir(folder):
                    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                    if files:
                        # ambil file pertama saja (karena by design hanya ada satu)
                        filename = files[0]
            except Exception:
                logger.warning(f"Gagal membaca file pada folder: {folder}", exc_info=True)
            
            contexts_list.append({"doc_id": doc_id, "filename": filename})

        return {"contexts": contexts_list}
    except Exception as e:
        logger.exception("Failed to list contexts")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/contexts/{doc_id}/retrieve")
def retrieve_for_doc(doc_id: str, q: str, n_results: int = 5):
    """Retrieve top-n relevant chunks for query limited to doc_id.
       Returns structured hits: list of {id, document, metadata, distance}.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="query parameter 'q' is required")
    try:
        hits = bot.get_relevant_context(q, doc_ids=[doc_id], n_results=n_results)
        if not hits:
            return {"found": False, "hits": []}
        return {"found": True, "hits": hits}
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/contexts/{doc_id}")
def delete_context(doc_id: str):
    """Delete all chunks for doc_id from vector store and remove saved files."""
    try:
        del_result = bot.delete_doc(doc_id)
        # remove saved folder if exists
        doc_folder = os.path.join(CONTEXTS_FOLDER, doc_id)
        if os.path.exists(doc_folder):
            shutil.rmtree(doc_folder)
        return {"status": "ok", "deleted": del_result}
    except Exception as e:
        logger.exception("Delete context failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Example body:
    {
      "user_input": "Apa poin utama pasal 2?",
      "doc_ids": ["doc-2025-01"],
      "n_results": 5,
      "model": "llama3",
      "system_instructions": "Jawab singkat dalam bahasa Indonesia."
    }
    """
    if not req.user_input or not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")

    try:
        model = req.model or bot.DEFAULT_LLM_MODEL
        resp = bot.generate_answer(
            user_input=req.user_input,
            doc_ids=req.doc_ids,
            model=model,
            n_results=req.n_results,
            system_instructions=req.system_instructions,
        )
        return resp
    except Exception as e:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
