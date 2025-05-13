# note_generation/soap_generator.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_together import Together

def generate_medical_notes_from_summarized_groups(summarized_groups: dict, together_api_key: str) -> str:
    """Generate SOAP notes from summarized groups."""
    prompt_template = """
    You are a medical assistant generating professional medical notes from summarized conversation excerpts.
    Here are the summarized excerpts for each SOAP section:
    - Subjective: {subjective}
    - Objective: {objective}
    - Assessment: {assessment}
    - Plan: {plan}
    Generate concise medical notes in the SOAP format:
    - Subjective: Summarize patient-reported symptoms and history in one sentence.
    - Objective: Summarize observations, test results, and findings in one sentence.
    - Assessment: State the diagnosis or impressions in one sentence.
    - Plan: List treatment recommendations, medications, and follow-ups in one sentence.
    """
    note_prompt = PromptTemplate(
        input_variables=["subjective", "objective", "assessment", "plan"],
        template=prompt_template
    )
    llm = Together(
        model="google/gemma-2-27b-it",
        together_api_key="38a11d9280e22f5b8c2e38385f133672f06cd405ca1f2cbfd7216183c451a33e",
        temperature=0.7,
        max_tokens=300
    )
    chain = LLMChain(llm=llm, prompt=note_prompt)
    try:
        notes = chain.run(
            subjective=" ".join(summarized_groups["Subjective"]),
            objective=" ".join(summarized_groups["Objective"]),
            assessment=" ".join(summarized_groups["Assessment"]),
            plan=" ".join(summarized_groups["Plan"])
        )
        return notes
    except Exception as e:
        return f"Error generating notes: {e}"