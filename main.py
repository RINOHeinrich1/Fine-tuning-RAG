import os
import faiss
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Configuration ---
EMBEDDING_MODEL_NAME = "models/esti-rag-ft"

if os.path.exists(EMBEDDING_MODEL_NAME):
    print(f"📂 Chargement du modèle fine-tuné depuis {EMBEDDING_MODEL_NAME}")
else:
    print(f"🌐 Aucun modèle fine-tuné trouvé. Chargement du modèle de base : {EMBEDDING_MODEL_NAME}")
    EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 3
BATCH_SIZE = 4
EPOCHS = 2
WARMUP_STEPS = 10

# --- Initialisation ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace "*" par ["http://localhost:5173"] ou ton domaine en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

documents = [
    "L'ESTI est une école supérieure privée située à Antananarivo, Madagascar.",
    "L'ESTI signifie École Supérieure des Technologies de l'Information.",
    "L'ESTI propose des formations dans les domaines de l'informatique, du développement logiciel, des réseaux, de la cybersécurité et du management des systèmes d'information.",
    "Les diplômes délivrés par l'ESTI sont homologués et reconnus par l'État malgache.",
    "L'ESTI met l'accent sur l'acquisition de compétences pratiques à travers des projets concrets et des stages en entreprise.",
    "En plus des formations initiales, l'ESTI propose aussi des formations modulaires qui débouchent sur des certificats professionnels.",
    "L'ESTI accompagne ses étudiants vers des carrières en freelancing ou dans des entreprises du secteur numérique.",
    "Le contact téléphonique de l'ESTI est 0330828086, 0340220452 ou 0320420452."
]

# --- FAISS Index ---
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
index = build_faiss_index(doc_embeddings)

def search_faiss(question: str, k=TOP_K):
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, k)
    return [documents[i] for i in indices[0]], distances[0].tolist()

def fine_tune_model(question, positive_docs, negative_docs):
    train_examples = []
    for doc in positive_docs:
        train_examples.append(InputExample(texts=[question, doc], label=1.0))
    for doc in negative_docs:
        train_examples.append(InputExample(texts=[question, doc], label=0.0))

    if not train_examples:
        print("⚠️ Aucun exemple pour fine-tuning.")
        return

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)

    # 🔍 Vérifier les paramètres entraînables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Nombre de paramètres entraînables : {trainable_params}")
    if trainable_params == 0:
        print("❌ Aucun paramètre à entraîner. Abandon.")
        return

    print(f"🏋️ Fine-tuning sur {len(train_examples)} exemples...")
    model.train()  # Assure le mode training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True
    )
    print("✅ Fine-tuning terminé.")

    # 💾 Sauvegarde temporaire et rechargement du modèle pour rafraîchir les poids
    model.save("models/esti-rag-ft")
    return SentenceTransformer("models/esti-rag-ft", device=DEVICE)

def evaluate_score_margin(model, question, positive_docs, negative_docs, device):
    model.eval()
    with torch.no_grad():
        q_emb = model.encode(question, convert_to_tensor=True, device=device)

        pos_scores = []
        for doc in positive_docs:
            d_emb = model.encode(doc, convert_to_tensor=True, device=device)
            pos_scores.append(torch.nn.functional.cosine_similarity(q_emb, d_emb, dim=0).item())

        neg_scores = []
        for doc in negative_docs:
            d_emb = model.encode(doc, convert_to_tensor=True, device=device)
            neg_scores.append(torch.nn.functional.cosine_similarity(q_emb, d_emb, dim=0).item())

    min_pos = min(pos_scores)
    max_neg = max(neg_scores)
    return min_pos, max_neg, pos_scores, neg_scores

def fine_tune_until_margin_respected(question, positive_docs, negative_docs,
                                     model, batch_size, epochs, warmup_steps, device,
                                     max_iterations=10):
    iteration = 0
    current_model = model

    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 Itération #{iteration} de fine-tuning...")

        train_examples = [
            InputExample(texts=[question, doc], label=1.0) for doc in positive_docs
        ] + [
            InputExample(texts=[question, doc], label=0.0) for doc in negative_docs
        ]

        if not train_examples:
            print("⚠️ Aucun exemple pour fine-tuning.")
            break

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(current_model)

        current_model.train()
        current_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True
        )
        print("✅ Fine-tuning terminé pour cette itération.")

        min_pos, max_neg, pos_scores, neg_scores = evaluate_score_margin(
            current_model, question, positive_docs, negative_docs, device
        )

        print(f"📈 Scores positifs : {['%.4f' % s for s in pos_scores]}")
        print(f"📉 Scores négatifs : {['%.4f' % s for s in neg_scores]}")
        print(f"✅ min(score_positif) = {min_pos:.4f}")
        print(f"❌ max(score_négatif) = {max_neg:.4f}")

        if min_pos > max_neg:
            print("🎯 Condition atteinte : tous les positifs sont mieux scorés que tous les négatifs.")
            break
        else:
            print("🔁 Encore des négatifs mieux scorés que des positifs. On continue...")

    current_model.save("models/esti-rag-ft")
    return SentenceTransformer("models/esti-rag-ft", device=device)

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str
    top_k: int = TOP_K

class FeedbackRequest(BaseModel):
    question: str
    positive_docs: List[str]
    negative_docs: List[str]

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "✅ RAG Webservice is running."}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        docs, scores = search_faiss(request.question, request.top_k)
        return {
            "question": request.question,
            "results": [{"doc": doc, "score": score} for doc, score in zip(docs, scores)]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/documents")
def get_all_documents():
    return {"documents": documents}

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    global model, doc_embeddings, index

    try:
        # 🔍 Avant fine-tuning : rechercher les résultats initiaux
        docs_before, scores_before = search_faiss(request.question)

        # 🏋️ Fine-tuning
        updated_model = fine_tune_until_margin_respected(
    request.question, request.positive_docs,request.negative_docs,model,BATCH_SIZE,EPOCHS,WARMUP_STEPS,DEVICE,
    max_iterations=10
)

        if updated_model is None:
            raise HTTPException(status_code=400, detail="Aucun exemple de fine-tuning fourni.")

        # 🔄 Mise à jour du modèle et de l'index
        model = updated_model
        doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        index = build_faiss_index(doc_embeddings)

        # 🔍 Après fine-tuning : rechercher à nouveau
        docs_after, scores_after = search_faiss(request.question)

        # 📊 Comparaison
        comparison = {
            "before": [{"doc": d, "score": s} for d, s in zip(docs_before, scores_before)],
            "after": [{"doc": d, "score": s} for d, s in zip(docs_after, scores_after)]
        }
        print(comparison)

        return {
            "message": "✅ Fine-tuning terminé et index mis à jour.",
            "comparison": comparison
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Lancement du serveur ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
