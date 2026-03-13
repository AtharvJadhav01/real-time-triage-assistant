from triage_assistant import TriageAssistant

assistant = TriageAssistant()

docs = [
    {
        "text": "Severe bleeding protocol: apply tourniquet immediately.",
        "type": "protocol",
        "tags": ["trauma", "bleeding"]
    },
    {
        "text": "Dental cavity treatment requires cleaning and filling.",
        "type": "protocol",
        "tags": ["dental"]
    },
    {
        "text": "If patient shows signs of shock, elevate legs and administer IV fluids.",
        "type": "protocol",
        "tags": ["shock"]
    }
]

assistant.ingest(docs)


query = """
Patient has deep leg wound from accident,
bleeding heavily and showing dizziness.
"""

response = assistant.query(query)

print(response)
