import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FAISS index and metadata paths
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "../data/mayo_faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "../data/mayo_metadata.npy")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index with error handling
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("FAISS index loaded successfully.")
else:
    raise FileNotFoundError(f"FAISS index file not found: {FAISS_INDEX_PATH}")

# Load metadata with error handling
if os.path.exists(METADATA_PATH):
    metadata = np.load(METADATA_PATH, allow_pickle=True)
    print("Metadata loaded successfully.")
else:
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

def search(query, top_k=5):
    """Search FAISS for similar diseases given a symptom query."""
    if not query.strip():
        print("Error: Query cannot be empty.")
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [
        (metadata[i], distances[0][idx]) for idx, i in enumerate(indices[0]) if i < len(metadata)
    ]
    
    return results

