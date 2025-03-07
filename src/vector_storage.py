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

    # Load scraped Mayo Clinic data with proper error handling
    try:
        with open("../data/mayo_disease_data.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Ensure the JSON is a dictionary and flatten it into a list of diseases
        if not isinstance(raw_data, dict):
            raise ValueError("Invalid JSON format: Expected a dictionary with letter keys.")

        diseases = []
        for letter, disease_list in raw_data.items():
            if isinstance(disease_list, list):  # Ensure it’s a list
                diseases.extend(disease_list)
            else:
                print(f"Skipping invalid section: {letter}")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Loaded {len(diseases)} disease entries.")

    # Extract and preprocess text
    text_chunks = []
    metadata = []  # Stores disease names for later retrieval

    for disease in diseases:
        if not isinstance(disease, dict):  # Ensure each entry is a dictionary
            print(f"Skipping invalid entry: {disease}")
            continue

        name = disease.get("disease", "Unknown Disease")
        symptoms = disease.get("symptoms", "Not Available")
        causes = disease.get("causes", "Not Available")
        treatment = disease.get("treatment", "Not Available")

        text = f"{symptoms} {causes} {treatment}".strip()

        # Skip completely unavailable data
        if text == "Not Available Not Available Not Available":
            continue

        text_chunks.append(text)
        metadata.append(name)

    print(f"Processed {len(text_chunks)} valid disease entries.")

    if not text_chunks:
        print("No valid diseases found. Exiting.")
        return

    # Embed the text
    try:
        embeddings = embed_text(text_chunks)
        print(f"Generated {embeddings.shape[0]} embeddings.")
    except Exception as e:
        print(f"Embedding Error: {e}")
        return

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    try:
        faiss.write_index(index, "../data/mayo_faiss.index")
        print("FAISS vector database stored.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
        return

    # Save metadata separately for lookup
    try:
        np.save("../data/mayo_metadata.npy", np.array(metadata, dtype=object))
        print("Metadata stored.")
    except Exception as e:
        print(f"Error saving metadata: {e}")

if __name__ == "__main__":
    store_embeddings()
