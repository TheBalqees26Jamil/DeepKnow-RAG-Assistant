import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("models/gemini-flash-lite-latest")


def generate_answer(query, contexts):
    """
    Generate answer using Gemini API with error handling.
    Returns safe fallback message on any failure.
    """
    
    
    if not contexts:
        return "I could not find relevant context in the knowledge base."


    context_text = "\n\n".join([c["text"] for c in contexts])

    prompt = f"""
You are a RAG assistant.

Answer ONLY using the provided context.

If the answer is not in the context, say:
"I could not find the answer in the knowledge base."

Do not use external knowledge.

Context:
{context_text}

Question:
{query}

Answer:
"""

    try:
        response = model.generate_content(prompt)
        
       
        if not response:
            print("Warning: Empty response from Gemini API")
            return "Unable to generate answer right now."
        
     
        if not hasattr(response, 'text') or response.text is None:
            print("Warning: Response has no text content")
            return "Unable to generate answer right now."
        
      
        answer = response.text.strip()
        if not answer:
            return "Unable to generate answer right now."
        
        return answer
        
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Unable to generate answer right now."