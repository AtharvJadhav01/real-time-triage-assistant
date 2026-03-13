"""
Data ingestion module.

Responsible for loading medical documents, disaster protocols,
and patient records into structured chunks.
"""

import json

def load_documents(file_path):
    """
    Load medical protocol or patient history documents.

    Expected format:
    [
        {"text": "...", "type": "protocol", "tags": ["trauma", "bleeding"]},
        {"text": "...", "type": "patient_history", "tags": ["diabetes"]}
    ]
    """
    with open(file_path, "r") as f:
        docs = json.load(f)

    return docs


def chunk_documents(documents, chunk_size=300):
    """
    Splits documents into smaller chunks to improve embedding retrieval.
    """
    chunks = []

    for doc in documents:
        text = doc["text"]
        for i in range(0, len(text), chunk_size):
            chunk = {
                "text": text[i:i+chunk_size],
                "type": doc["type"],
                "tags": doc.get("tags", [])
            }
            chunks.append(chunk)

    return chunks