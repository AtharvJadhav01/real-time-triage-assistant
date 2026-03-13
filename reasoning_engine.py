"""
Reasoning engine using LLM to recommend triage actions.
"""
import requests

class ReasoningEngine:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"

    def generate_action(self, query, context_docs):

        context = "\n".join([doc["text"] for doc in context_docs])

        prompt = f"""
You are an emergency triage assistant.

Patient Situation:
{query}

Relevant Protocols:
{context}

Provide:
1. Severity Level
2. Immediate Action
3. Required Medical Team
4. Equipment Needed
"""

        data = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.url, json=data)

        return response.json()["response"]