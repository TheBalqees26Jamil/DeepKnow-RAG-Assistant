import google.generativeai as genai
from config import GEMINI_API_KEY


genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("models/gemini-flash-lite-latest")


def generate_answer(query, contexts):
    

   
    context_text = "\n\n".join([c["text"] for c in contexts])

    prompt = f"""
You are a helpful AI assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say: "I don't know based on the given data."

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = model.generate_content(prompt)

    return response.text



