# src/searcher.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
import os


class MedicalSearcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name  # comparing models
        self.index = None
        self.df = None

    def build_index(self, df, text_column='cleaned_transcription'):
        """
        Generates embeddings and builds FAISS index.
        """
        self.df = df.reset_index(drop=True)
        print("Generating embeddings...")
        texts = df[text_column].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Index built with {self.index.ntotal} documents.")
        return embeddings

    def search(self, query, k=5):
        """
        Searches for k most similar documents to the query.
        """
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'score': float(score),
                'specialty': self.df.iloc[idx]['medical_specialty'],
                'text_preview': self.df.iloc[idx]['cleaned_transcription'][:200]
            })
        return results

    def precision_at_k(self, query, true_specialty, k=5):
        """
        Calculates Precision@K for a given query.
        """
        results = self.search(query, k=k + 1)
        results = results[1:]
        relevant = sum(1 for r in results if r['specialty'] == true_specialty)
        return relevant / k

    def save(self, path=None):
        if path is None:
            path = f"data/faiss_index_{self.model_name.replace('/', '_')}"
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        self.df.to_csv(f"{path}/documents.csv", index=False)
        print(f"Index saved to {path}")

    def load(self, path=None):
        if path is None:
            path = f"data/faiss_index_{self.model_name.replace('/', '_')}"
        self.index = faiss.read_index(f"{path}/index.faiss")
        self.df = pd.read_csv(f"{path}/documents.csv")
        print(f"Index loaded: {self.index.ntotal} documents")


def evaluate_search(searcher, df, k_values=(1, 3, 5), n_queries=100):
    """
    Evaluates search quality using Precision@K.
    """
    sample = df.sample(n=n_queries, random_state=42)
    results = {}

    for k in k_values:
        precisions = []
        for _, row in sample.iterrows():
            p = searcher.precision_at_k(
                row['cleaned_transcription'][:500],
                row['medical_specialty'],
                k=k
            )
            precisions.append(p)
        results[f'Precision@{k}'] = np.mean(precisions)
        print(f"Precision@{k}: {results[f'Precision@{k}']:.4f}")

    return results