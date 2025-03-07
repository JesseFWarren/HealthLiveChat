import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text_chunks):
    """Convert text chunks into vector embeddings"""
    return model.encode(text_chunks, convert_to_numpy=True)

def store_embeddings():
    """Loads Mayo Clinic disease data, embeds it, and stores in FAISS."""
    
    # Load scraped Mayo Clinic data
    with open("../data/mayo_disease_data.json", "r") as f:
        diseases = json.load(f)

    print(f"Loaded {len(diseases)} diseases.")

    # Extract and preprocess text
    text_chunks = []
    metadata = []  # Stores disease names for later retrieval
    
    for disease in diseases:
        name = disease["disease"]
        text = f"{disease['symptoms']} {disease['causes']} {disease['treatment']}"
        
        # Skip empty text entries
        if text.strip() == "Not Available Not Available Not Available":
            continue
        
        text_chunks.append(text)
        metadata.append(name)

    print(f"Processed {len(text_chunks)} valid disease entries.")

    # Embed the text
    embeddings = embed_text(text_chunks)
    print(f"Generated {embeddings.shape[0]} embeddings.")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, "../data/mayo_faiss.index")
    print("FAISS vector database stored.")

    # Save metadata separately for lookup
    np.save("../data/mayo_metadata.npy", np.array(metadata, dtype=object))
    print("Metadata stored.")

if __name__ == "__main__":
    store_embeddings()
