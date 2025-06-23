# trainer.py
import os
import faiss
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from knowledge import CORPUS

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_MODEL_NAME = "models/esti-rag-ft" if os.path.exists("models/esti-rag-ft") else "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
BATCH_SIZE = 4
EPOCHS = 2
WARMUP_STEPS = 10
FAISS_DIR = "faiss_data"
DOCUMENTS_FILE = os.path.join(FAISS_DIR, "documents.json")

# Charger modèle
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

documents = CORPUS
# Charger index FAISS si possible
INDEX_PATH = os.path.join(FAISS_DIR, "esti.index")
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = None

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index, embeddings, documents, dir_path=FAISS_DIR):
    os.makedirs(dir_path, exist_ok=True)
    faiss.write_index(index, f"{dir_path}/esti.index")
    np.save(f"{dir_path}/esti_embeddings.npy", embeddings)
    with open(f"{dir_path}/documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

def fine_tune(question, positive_docs, negative_docs, corpus_docs):
    train_examples = [
        InputExample(texts=[question, doc], label=1.0) for doc in positive_docs
    ] + [
        InputExample(texts=[question, doc], label=0.0) for doc in negative_docs
    ]

    if not train_examples:
        return False, "Aucun exemple fourni."

    dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    loss_fn = losses.CosineSimilarityLoss(model)

    model.train()
    model.fit(
        train_objectives=[(dataloader, loss_fn)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True
    )

    # Sauvegarde modèle + nouvel index
    model.save("models/esti-rag-ft")
    new_model = SentenceTransformer("models/esti-rag-ft", device=DEVICE)
    new_embeddings = new_model.encode(corpus_docs, convert_to_numpy=True, normalize_embeddings=True)
    new_index = build_faiss_index(new_embeddings)
    save_faiss_index(new_index, new_embeddings, corpus_docs)

    global index, documents
    index = new_index
    documents = corpus_docs

    return True, "✅ Fine-tuning terminé avec succès."

def load_or_create_index(corpus_docs):
    global index, documents

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCUMENTS_FILE):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
    else:
        documents = corpus_docs
        if not documents or len(documents) == 0:
            raise ValueError("Le corpus de documents est vide, impossible de créer l'index FAISS.")
        embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings shape inattendue : {embeddings.shape}, attendu (n_samples, dim)")
        index = build_faiss_index(embeddings)
        save_faiss_index(index, embeddings, documents)

    return index, documents

# Charger documents + index au démarrage (ajouter un paramètre corpus par défaut si besoin)
index, documents = load_or_create_index(documents)

def search_faiss(question, k=TOP_K):
    global index, documents

    if index is None or not documents:
        return [], []

    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs, distances[0].tolist()
