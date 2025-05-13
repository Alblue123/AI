# rag/chunk_grouping.py
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def rule_based_preprocess(conversation: str) -> list:
    """Tag conversation lines with potential SOAP sections."""
    tagged_lines = []
    lines = conversation.split("\n")
    for line in lines:
        line_lower = line.lower().strip()
        if not line_lower:
            continue
        if line.startswith("Patient:") and any(word in line_lower for word in ["feel", "have", "pain", "since"]):
            tagged_lines.append((line, "Subjective"))
        elif any(word in line_lower for word in ["temperature", "measured", "observed", "vitals"]):
            tagged_lines.append((line, "Objective"))
        elif any(word in line_lower for word in ["infection", "diagnosis", "condition", "seems"]):
            tagged_lines.append((line, "Assessment"))
        elif any(word in line_lower for word in ["prescribe", "follow-up", "treatment", "recommend"]):
            tagged_lines.append((line, "Plan"))
        else:
            tagged_lines.append((line, "Unknown"))
    return tagged_lines

def group_chunks_by_soap(conversation: str, retriever) -> dict:
    """Group conversation chunks by SOAP section."""
    tagged_lines = rule_based_preprocess(conversation)
    
    # Split conversation into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = [Document(page_content=conversation)]
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="emilyalsentzer/Bio_ClinicalBERT")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Define queries
    soap_queries = {
        "Subjective": "What symptoms or history did the patient report?",
        "Objective": "What observations or test results were mentioned?",
        "Assessment": "What diagnosis or impressions were stated?",
        "Plan": "What treatments or follow-ups were recommended?"
    }
    
    # Group chunks
    soap_groups = {"Subjective": [], "Objective": [], "Assessment": [], "Plan": []}
    for section, query in soap_queries.items():
        relevant_docs = retriever.get_relevant_documents(query)
        for doc in relevant_docs:
            chunk_content = doc.page_content
            for line, tagged_section in tagged_lines:
                if line in chunk_content and (tagged_section == section or tagged_section == "Unknown"):
                    soap_groups[section].append(chunk_content)
                    break
            else:
                soap_groups[section].append(chunk_content)
    
    # Deduplicate chunks
    for section in soap_groups:
        soap_groups[section] = list(set(soap_groups[section]))
    
    return soap_groups