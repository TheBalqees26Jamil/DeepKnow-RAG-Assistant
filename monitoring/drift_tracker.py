import json
import os
from datetime import datetime

LOG_FILE = "monitoring/drift_logs.json"


def log_metrics(query: str, metrics: dict):

    print("DRIFT LOGGING STARTED")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "groundedness": metrics.get("groundedness_score", 0),
        "relevance": metrics.get("relevance_score", 0),
        "overall": metrics.get("overall_score", 0),
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("DRIFT ENTRY SAVED")