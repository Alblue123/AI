# SOAP Medical Notes Generator

The **SOAP Medical Notes Generator** is a Streamlit-based application designed to automate the creation of professional SOAP (Subjective, Objective, Assessment, Plan) notes from patient-doctor conversation dialogues. By leveraging advanced natural language processing techniques, including Retrieval-Augmented Generation (RAG) and extractive summarization, the app processes preprocessed dialogue data to produce concise, accurate medical notes, reducing the administrative burden on healthcare professionals.

## Highlights

- **Automated SOAP Note Generation**: Converts preprocessed patient-doctor dialogues into concise SOAP notes using RAG and extractive summarization.
- **RAG-Based Chunk Grouping**: Employs Retrieval-Augmented Generation to group conversation chunks into SOAP sections, guided by rule-based preprocessing.
- **Extractive Summarization**: Refines grouped chunks using TextRank to produce concise, relevant summaries for each SOAP section.
- **Interactive Streamlit Interface**: Provides a user-friendly UI to upload CSV files, select encounters, and generate or query SOAP notes.
- **Scalable and Extensible**: Designed to handle large datasets and support future enhancements like model fine-tuning or additional NLP techniques.

## Detailed Architecture

The Medical Notes Generator is structured as a modular pipeline that processes CSV inputs (produced by a preprocessing module) to generate SOAP notes and collect fine-tuning data. Below is the detailed architecture, organized by module and functionality.

### Pipeline Workflow

1. **Preprocessing**:
   - The `preprocess` module produces a CSV with clean, role-corrected dialogue.

2. **RAG Initialization**:
   - `rag.rag_utils.initialize_rag` creates a FAISS vector store, retriever, and QA chain from `fixed_role_dialogue`.

3. **Chunk Grouping**:
    - Applies rule-based preprocessing to tag lines.
    - Uses RAG to retrieve and group chunks into SOAP sections (Subjective, Objective, Assessment, Plan).

4. **Extractive Summarization**:
   - Summarizes each SOAP group using TextRank, producing concise sentences.

5. **SOAP Note Generation**:
   - Generates professional SOAP notes.
