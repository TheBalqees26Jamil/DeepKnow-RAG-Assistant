import re


BLOCKED_PATTERNS = [
    r"ignore previous instructions",
    r"bypass",
    r"reveal system prompt",
    r"hack",
    r"malware",
    r"steal",
    r"api key",
]


def is_safe_query(query):

    query = query.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query):
            return False

    return True