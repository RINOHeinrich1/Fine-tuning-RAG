import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
# --- Config ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 3
BATCH_SIZE = 4
EPOCHS = 2
WARMUP_STEPS = 10

# --- 1. Chargement du modèle et des documents ---
print("📦 Chargement du modèle...")
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

# --- 2. Création de l'index FAISS ---
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
index = build_faiss_index(doc_embeddings)

#--- 3. Sauvegarde de l'index FAISS ---
def save_faiss_index(index, embeddings, dir_path="faiss_data", index_filename="esti.index", emb_filename="esti_embeddings.npy"):
    """
    Sauvegarde l'index FAISS et les embeddings dans un dossier donné.

    Args:
        index: L'objet FAISS (ex: IndexFlatIP)
        embeddings: Les vecteurs de documents (numpy array)
        dir_path: Dossier de destination (sera créé s'il n'existe pas)
        index_filename: Nom du fichier index FAISS
        emb_filename: Nom du fichier des embeddings .npy
    """
    os.makedirs(dir_path, exist_ok=True)

    faiss.write_index(index, os.path.join(dir_path, index_filename))
    np.save(os.path.join(dir_path, emb_filename), embeddings)

    print(f"✅ Index FAISS sauvegardé dans {os.path.join(dir_path, index_filename)}")
    print(f"✅ Embeddings sauvegardés dans {os.path.join(dir_path, emb_filename)}")
    
# --- 3. Recherche FAISS ---
def search_faiss(question, k=TOP_K):
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs, distances[0]

# --- 4. Fine-tuning ---
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

try:
    import readline
except ImportError:
    pass  # Ignore if not available

try:
    # --- 5. Boucle interactive ---
    print("\n--- 🔁 RAG interactif avec feedback utilisateur ---")
    print("Tape 'exit' pour quitter.\n")

    while True:
        question = input("❓ Question : ")
        if question.lower() in ['exit', 'quit']:
            break

        retrieved_docs, scores = search_faiss(question)
        print("\n📚 Documents récupérés :")
        for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
            print(f"[{i}] (score={score:.4f}) {doc}")

        feedback = input("🧠 Indique les indices des docs pertinents (ex: 0 2), ou tape 'none' si aucun : ").strip().lower()

        if feedback == "":
            print("⏭️ Pas de feedback, pas de fine-tuning cette fois.\n")
            continue

        elif feedback == "none":
            # Afficher tous les documents pour un choix manuel
            print("\n📋 Tous les documents disponibles :")
            for i, doc in enumerate(documents):
                print(f"[{i}] {doc}")

            confirm = input("\n✅ Indique les indices des bons documents (positifs), ou rien pour ne rien faire : ").strip()
            if confirm == "":
                print("⏭️ Aucun document jugé pertinent. Skip.\n")
                continue

            try:
                good_indices = list(map(int, confirm.split()))
            except:
                print("⚠️ Entrée invalide, recommence.")
                continue

            positive_docs = [documents[i] for i in good_indices if 0 <= i < len(documents)]
            negative_docs = [documents[i] for i in range(len(documents)) if i not in good_indices]

        else:
            try:
                good_indices = list(map(int, feedback.strip().split()))
            except:
                print("⚠️ Entrée invalide, recommence.")
                continue

            positive_docs = [retrieved_docs[i] for i in good_indices if 0 <= i < len(retrieved_docs)]
            negative_docs = [retrieved_docs[i] for i in range(len(retrieved_docs)) if i not in good_indices]

        print("🔍 Avant fine-tuning :", search_faiss(question))
        updated_model = fine_tune_model(question, positive_docs, negative_docs)

        if updated_model:
            model = updated_model
            doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
            index = build_faiss_index(doc_embeddings)
            save_faiss_index(index,doc_embeddings)
            print("🔁 Index FAISS mis à jour avec les nouveaux embeddings.")
            print("🔍 Après fine-tuning :", search_faiss(question))
        else:
            print("⚠️ Fine-tuning non appliqué.")

        print()

    print("👋 Fin du programme.")
except KeyboardInterrupt:
    print("\n👋 Fin du programme (interruption clavier détectée).")