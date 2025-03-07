import os
import faiss
import numpy as np
from openai import OpenAI
from src.retrieval import search

# OpenAI API key and Client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY as an environment variable.")
client = OpenAI(api_key=OPENAI_API_KEY)

def ask_medical_chatbot(user_query):
    """
    Retrieves relevant Mayo Clinic disease context and queries GPT-4.
    """
    try:
        retrieved_diseases = search(user_query)

        if not retrieved_diseases:
            return "I couldn't find relevant medical information. Try rephrasing your symptoms."

        context = "\n".join(retrieved_diseases)
        prompt = f"""
        You are a medical assistant trained on Mayo Clinic data. Use the provided medical context to suggest possible condition, treatment options, 
        and guide the user, but do NOT provide a formal diagnosis. Instead, suggest consulting a healthcare professional. Keep responses to a paragrph.

        Context:
        {context}

        User's Symptoms: {user_query}
        Answer:
        """

        # Query GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical assistant providing guidance based on Mayo Clinic data. You do NOT diagnose users."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    user_input = input("Describe your symptoms: ")
    print(ask_medical_chatbot(user_input))
