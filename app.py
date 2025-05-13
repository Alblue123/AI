# app/main.py
import streamlit as st
from preprocess.data_loader import load_dataframe
from rag.rag_utils import initialize_rag
from rag.chunk_grouping import group_chunks_by_soap
from summarization.extractive import summarize_soap_groups
from note_generation.soap_generator import generate_medical_notes_from_summarized_groups
from note_generation.data_saver import save_summarized_groups

def main():
    st.title("Medical Notes Generator")
    
    # Load API key (replace with .env in production)
    TOGETHER_API_KEY = "38a11d9280e22f5b8c2e38385f133672f06cd405ca1f2cbfd7216183c451a33e"
    
    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'conversation_text' not in st.session_state:
        st.session_state.conversation_text = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # File upload for DataFrame
    uploaded_file = st.file_uploader("Upload DataFrame (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        df = load_dataframe(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.success("DataFrame loaded successfully!")
    
    # Select encounter_id
    if st.session_state.df is not None:
        encounter_ids = st.session_state.df['encounter_id'].tolist()
        selected_encounter = st.selectbox("Select Encounter ID", [""] + encounter_ids)
        
        if selected_encounter:
            row = st.session_state.df[st.session_state.df['encounter_id'] == selected_encounter].iloc[0]
            fixed_role_dialogue = row['fixed_role_dialogue']
            reference_note = row.get('note', None)
            
            # Initialize system
            if st.button("Initialize System"):
                with st.spinner("Initializing system..."):
                    st.session_state.qa_chain, st.session_state.conversation_text, st.session_state.retriever = \
                        initialize_rag(fixed_role_dialogue, TOGETHER_API_KEY)
                    if st.session_state.qa_chain:
                        st.success("System initialized successfully!")
                    else:
                        st.error("Failed to initialize system.")
    
    # Tabs
    tab1, tab2 = st.tabs(["Question Answering", "SOAP Notes Generation"])
    
    with tab1:
        st.header("Question Answering")
        if st.session_state.qa_chain:
            question = st.text_input("Ask a question about the conversation:")
            if st.button("Get Answer"):
                try:
                    with st.spinner("Generating answer..."):
                        result = st.session_state.qa_chain({"query": question})
                        st.write("**Answer:**")
                        st.write(result["result"])
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append({"role": "assistant", "content": result["result"]})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.info("Please initialize the system first")
    
    with tab2:
        st.header("SOAP Notes Generation")
        if st.session_state.conversation_text and st.session_state.retriever:
            if st.button("Generate SOAP Notes"):
                try:
                    with st.spinner("Generating SOAP notes..."):
                        # Group and summarize
                        soap_groups = group_chunks_by_soap(st.session_state.conversation_text, st.session_state.retriever)
                        summarized_groups = summarize_soap_groups(soap_groups)
                        
                        # Save for fine-tuning
                        save_summarized_groups(selected_encounter, st.session_state.conversation_text, summarized_groups, reference_note)
                        
                        # Generate notes
                        notes = generate_medical_notes_from_summarized_groups(summarized_groups, TOGETHER_API_KEY)
                        st.write("**Generated SOAP Notes:**")
                        st.write(notes)
                except Exception as e:
                    st.error(f"Error generating notes: {e}")
        else:
            st.info("Please initialize the system first")

if __name__ == "__main__":
    main()