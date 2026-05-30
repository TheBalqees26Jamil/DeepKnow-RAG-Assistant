# DeepKnow RAG Assistant

A fully modular Retrieval-Augmented Generation (RAG) system designed to answer questions from your own knowledge base using Deep Learning concepts. Built with **FastAPI** backend, **FAISS** for vector search, **Google Gemini** for generation, and **Streamlit** for the UI.

---

##  Features

-  **Ingest & Chunk** — Load `.txt` files, clean text, and split into overlapping chunks
-  **Embed & Store** — Generate dense embeddings using `all-MiniLM-L6-v2` and store them with FAISS
-  **Semantic Retrieval** — Retrieve top-k relevant chunks using cosine similarity via FAISS
-  **LLM Generation** — Generate grounded answers using Google Gemini (`gemini-flash-lite-latest`)
-  **Safety Guardrails** — Block malicious or jailbreak-style queries before processing
- **Built-in Evaluation** — Compute groundedness, relevance, precision@k, recall@k, and overall score
-  **Stunning UI** — Dark-themed Streamlit interface with neon aesthetics

---

##  Project Structure

```
DeepKnow_RAG_Assistant/
│
├── app.py                      # Streamlit frontend (Home + Chat pages)
├── rag_pipeline.py             # End-to-end RAG pipeline orchestrator
├── .env                        # API keys & secrets (not committed)
├── .env.example                # Template for environment variables
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignored files/folders
├── README.md                   # This file
├── lucid.jpg                   # Image for the landing page
│
├── api/                        # FastAPI backend
│   ├── main.py                 # API entry point (FastAPI app)
│   ├── routes.py               # API endpoints (/ask, /health)
│   └── schemas.py              # Pydantic request/response models
│
├── data/
│   ├── processed/              # Raw .txt knowledge base files
│   └── embeddings.pkl          # Serialized vector store
│
├── ingestion/
│   ├── loader.py               # Load .txt documents from disk
│   └── chunker.py              # Clean & split text into chunks
│
├── embeddings/
│   └── embedder.py             # Generate & save embeddings
│
├── retrieval/
│   └── retriever.py            # FAISS index builder & semantic search
│
├── llm/
│   └── llm_client.py           # Google Gemini API client
│
├── evaluation/
│   └── metrics.py              # RAG evaluation metrics
│
└── safety/
    └── guardrails.py           # Query safety filtering
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/DeepKnow_RAG_Assistant.git
cd DeepKnow_RAG_Assistant
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download `all-MiniLM-L6-v2` on first run (~80MB).

### 4. Set up environment variables

```bash
cp .env.example .env
```

Then edit `.env` and add your **Google Gemini API key**:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

Get your key from: [Google AI Studio](https://aistudio.google.com/app/apikey)

---

##  Preparing Your Knowledge Base

1. Place your `.txt` files inside `data/processed/`
2. Each file should contain plain text about your domain (e.g., Deep Learning notes)

Example:
```
data/processed/
├── activation_functions.txt
├── backpropagation.txt
├── cnn_architectures.txt
└── optimizers.txt
```

---

## Running the System

The system now runs in **two separate processes**: a **FastAPI backend** and a **Streamlit frontend**.

### Step 1: Chunk & Embed (One-time setup)
```bash
python ingestion/chunker.py       # Optional: preview chunks
python embeddings/embedder.py     # Generate embeddings & save to data/embeddings.pkl
```

### Step 2: Start the Backend API
```bash
uvicorn api.main:app --reload 
```
> The API will be available at: `http://127.0.0.1:8000`

**Verify it's running:**
```bash
curl http://127.0.0.1:8000/health
```
Expected response:
```json
{"status": "healthy", "service": "DeepKnow RAG API"}
```

### Step 3: Start the Frontend (in a new terminal)
```bash
streamlit run app.py
```
> The app will open at: `http://localhost:8501`

---

## API Endpoints

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/` | GET | API status | — | `{"message": "DeepKnow RAG API is running"}` |
| `/health` | GET | Health check | — | `{"status": "healthy", "service": "DeepKnow RAG API"}` |
| `/ask` | POST | Ask a question | `{"query": "string", "show_chunks": false}` | `{"answer": "...", "retrieved_chunks": [...], "evaluation": {...}}` |

---

## Evaluation Journey

The evaluation system went through a **3-stage evolution** to achieve accurate, semantic-based assessment:

---

###  Stage 1 — Basic Evaluation (Initial)

**Approach:** Simple keyword overlap using `hash(word)` inside `simple_embed()`

**Problems:**
- No real semantic understanding
- Only literal word comparison
- Any paraphrasing reduced the score
- Results were low and inaccurate

**Scores:**
```
groundedness_score: 0.385
relevance_score:    0.440
overall_score:      0.608
```

**Why so low?**
- Evaluation was primitive (keyword-based)
- Gemini sometimes answered from general knowledge
- No true semantic comprehension

---

###  Stage 2 — Prompt Constraining

**Improvement:** Enhanced the RAG prompt in `llm/llm_client.py` to force Gemini to use **only** retrieved chunks.

**Prompt change:**
```
Answer ONLY using the provided context.
Do not use outside knowledge.
```

**What improved:**
- Groundedness increased significantly
- Hallucination decreased
- Gemini became "context-locked"

**Scores:**
```
groundedness_score: 0.815
```

> ⚠️ **Important Note:** While the score jumped to 0.815, the evaluation itself was **still primitive**. The high score reflected forced word overlap, not true semantic understanding. The metric was "overly optimistic" and not semantically accurate.

---

### Stage 3 — Semantic Evaluation (Current)

**Upgrade:** Replaced `simple_embed()` with real semantic embeddings using:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**What changed:**
- System now understands **semantic meaning**
- Sentences are compared **semantically**, not literally
- Measures **true similarity** between query, context, and answer
- Scores are more realistic and accurate

**Final Scores:**
```
groundedness_score: 0.655
relevance_score:    0.690
overall_score:      0.781
```

>  **Why did groundedness drop from 0.815 → 0.655?**
> 
> Because Stage 3 uses **real semantic evaluation**, not forced word overlap. The 0.655 score is **more honest and accurate** — it measures true meaning similarity, not just keyword matching.

---

### Summary of the Journey

| Stage | Problem | Solution | Groundedness | Evaluation Quality |
|-------|---------|----------|--------------|-------------------|
|  Basic | Keyword-only comparison | Built initial metrics | 0.385 |  Primitive |
|  Prompt Constraining | Gemini uses outside knowledge | Forced context-only answers | 0.815 | Overly optimistic |
|  Semantic | Evaluation not truly semantic | `sentence-transformers` | 0.655 | ✅ Accurate & Realistic |

---

###  Current Evaluation Metrics

The system now includes these automatic evaluation metrics:

| Metric | Description |
|--------|-------------|
| `answer_length_score` | Penalizes overly short answers |
| `groundedness_score` | **Semantic** similarity between answer & retrieved context |
| `relevance_score` | **Semantic** similarity between query & answer |
| `retrieval_precision` | Precision@K of retrieved chunks |
| `retrieval_recall` | Recall@K of retrieved chunks |
| `overall_score` | Average of all available metrics |

---

###  How to Run Evaluation

```python
from evaluation.metrics import evaluate_rag

results = evaluate_rag(
    query="What is ReLU?",
    retrieved_chunks=[chunk1, chunk2],
    answer="ReLU is a rectified linear unit...",
    relevant_chunks=[ground_truth_chunk]  # Optional
)

print(results)
```

---

### Final Result

The system transformed from:

```
❌ Keyword-based evaluation
```

to:

```
✅ Semantic embedding-based evaluation
```

**Benefits achieved:**
-  More accurate evaluation
-  Realistic groundedness measurement
-  Semantic relevance scoring
-  Reduced impact of paraphrasing on scores
-  Honest assessment of RAG quality

---

##  Safety & Guardrails

Blocked query patterns include:
- `ignore previous instructions`
- `bypass`, `hack`, `malware`, `steal`
- `reveal system prompt`, `api key`

If a query is flagged, the app responds with:
> {"detail": "Blocked unsafe query."}
And the frontend displays:
❌ Error 400: Blocked unsafe query

---

##  Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **LLM** | Google Gemini (`gemini-flash-lite-latest`) |
| **Language** | Python 3.10+ |

---

##  Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key for generation |

---

## Future Improvements

- [ ] Add support for PDF & Markdown ingestion
- [ ] Implement hybrid search (sparse + dense)
- [ ] Add conversation memory / chat history
- [ ] Deploy API to cloud (Render, Railway, or AWS)
- [ ] Deploy Streamlit to Hugging Face Spaces or Streamlit Cloud
- [ ] Add user feedback loop (thumbs up/down on answers)
- [ ] Add Docker containerization for portable deployment
- [ ] Add API authentication (API keys or OAuth2)


---

> **"Knowledge is only valuable when it can be retrieved."** — DeepKnow 

---


