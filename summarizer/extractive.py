# summarization/extractive.py
from summa import summarizer
import nltk
nltk.download('punkt')

def extractive_summarize_group(group: list, max_sentences: int = 2) -> list:
    """Apply extractive summarization to a group of chunks."""
    if not group:
        return []
    text = " ".join(group)
    summary = summarizer.summarize(text, ratio=0.5, split=True)
    return summary[:max_sentences]

def summarize_soap_groups(soap_groups: dict) -> dict:
    """Summarize each SOAP group."""
    summarized_groups = {}
    for section, chunks in soap_groups.items():
        summarized_groups[section] = extractive_summarize_group(chunks)
        if not summarized_groups[section]:
            summarized_groups[section] = ["No relevant information provided."]
    return summarized_groups