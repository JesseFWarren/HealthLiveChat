import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("../data/mayo_faiss.index")
metadata = np.load("../data/mayo_metadata.npy", allow_pickle=True)

def search(query, top_k=5):
    """Search FAISS for similar diseases given a symptom query."""
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    results = [(metadata[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    
    return results

# Example search
query = "fever, cough, and difficulty breathing"
results = search(query)

print("\nSearch Results:")
for disease, score in results:
    print(f"{disease} (Score: {score:.4f})")
