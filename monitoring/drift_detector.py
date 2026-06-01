import json
import time
import os

LOG_FILE = "monitoring/drift_logs.json"

def log_drift(query, answer, contexts, evaluation):
    drift_entry = {
        "timestamp": time.time(),
        "query": query,
        "answer": answer,
        "groundedness": evaluation.get("groundedness_score", 0),
        "relevance": evaluation.get("relevance_score", 0),
    }

    os.makedirs("monitoring", exist_ok=True)

    
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
    else:
        data = []

    data.append(drift_entry)

    #
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)