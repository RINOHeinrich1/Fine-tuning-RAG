from fastapi import APIRouter, HTTPException
from .schemas import QuestionRequest, FeedbackRequest
from model.embedder import load_model
from model.faiss_index import build_faiss_index, search
from model.documents import get_documents
from model.fine_tuning import fine_tune_until_margin_respected
import numpy as np
from config import BATCH_SIZE, EPOCHS, WARMUP_STEPS, DEVICE

router = APIRouter()

model = load_model()
documents = get_documents()
doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
index = build_faiss_index(doc_embeddings)

@router.get("/")
def root():
    return {"message": "✅ RAG Webservice is running."}

@router.get("/documents")
def list_documents():
    return {"documents": documents}

@router.post("/ask")
def ask(request: QuestionRequest):
    try:
        results, scores = search(index, model, documents, request.question, request.top_k)
        return {
            "question": request.question,
            "results": [{"doc": d, "score": s} for d, s in zip(results, scores)]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
def feedback(request: FeedbackRequest):
    global model, doc_embeddings, index

    try:
        before_docs, before_scores = search(index, model, documents, request.question)
        model = fine_tune_until_margin_respected(
            request.question, request.positive_docs, request.negative_docs,
            model, BATCH_SIZE, EPOCHS, WARMUP_STEPS, DEVICE,30
        )
        doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        index = build_faiss_index(doc_embeddings)
        after_docs, after_scores = search(index, model, documents, request.question)

        return {
            "message": "✅ Fine-tuning terminé et index mis à jour.",
            "comparison": {
                "before": [{"doc": d, "score": s} for d, s in zip(before_docs, before_scores)],
                "after": [{"doc": d, "score": s} for d, s in zip(after_docs, after_scores)]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
