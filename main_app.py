import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Medical Transcription Intelligence System", layout="wide")

# ============== CACHED RESOURCE LOADERS ==============

@st.cache_resource
def load_preprocessor():
    from src.preprocessor import MedicalPreprocessor
    return MedicalPreprocessor()

@st.cache_resource
def load_classifier():
    from src.utils import load_artifacts
    return load_artifacts(name="linear_svc_v1_no_surgery")

@st.cache_resource
def load_setfit():
    from src.setfit_classifier import load_setfit_model
    return load_setfit_model(name="setfit_v1_lemmatized_no_surgery")

@st.cache_resource
def load_searcher():
    from src.searcher import MedicalSearcher
    searcher = MedicalSearcher(model_name="all-MiniLM-L6-v2")
    searcher.load("data/faiss_index_all-MiniLM-L6-v2_cleaned_transcription")
    return searcher

@st.cache_resource
def load_dataset():
    return pd.read_csv("data/cleaned_data_cache.csv")

with st.spinner("Loading models..."):
    preprocessor = load_preprocessor()
    tfidf_pipeline = load_classifier()
    setfit_model, setfit_label_encoder = load_setfit()
    searcher = load_searcher()
    df = load_dataset()

st.title("Medical Transcription Intelligence System")
st.caption("Classification, semantic search, and information extraction from medical transcriptions")

tab1, tab2, tab3 = st.tabs(["Specialty Classification", "Semantic Search", "Entity Extraction"])

# ============== TAB 1: CLASSIFICATION ==============
with tab1:
    st.subheader("Medical Specialty Classification")
    text_input_clf = st.text_area("Medical transcription:", height=200, key="clf_input")
    model_choice = st.radio("Model:", ["TF-IDF + LinearSVC", "SetFit (few-shot)"], horizontal=True)

    if st.button("Classify"):
        if not text_input_clf.strip():
            st.warning("Please enter a transcription text.")
        else:
            with st.spinner("Classifying..."):
                cleaned = preprocessor.clean_text(text_input_clf)

                if model_choice == "TF-IDF + LinearSVC":
                    prediction = tfidf_pipeline.predict([cleaned])[0]
                else:
                    pred_idx = setfit_model.predict([cleaned])[0]
                    prediction = setfit_label_encoder.inverse_transform([pred_idx])[0]

            st.success(f"Predicted specialty: **{prediction}**")

            st.markdown("**Source — similar examples from the dataset for this specialty:**")
            examples = df[df['medical_specialty'] == prediction].head(3)
            for idx, row in examples.iterrows():
                with st.expander(f"Example from MTSamples dataset ({prediction})"):
                    st.write(row['transcription'][:500] + "...")

# ============== TAB 2: SEMANTIC SEARCH ==============
with tab2:
    st.subheader("Semantic Search over Medical Transcriptions")
    query = st.text_input("Search query:")
    k = st.slider("Number of results:", min_value=1, max_value=10, value=5)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                cleaned_query = preprocessor.clean_text(query)
                results = searcher.search(cleaned_query, k=k)

            st.markdown(f"**Found {len(results)} results:**")
            for i, r in enumerate(results, 1):
                with st.expander(f"#{i} — {r['specialty']} (similarity score: {r['score']:.3f})"):
                    st.write(r['text_preview'])
                    st.caption(f"Source: document from the MTSamples dataset, specialty: {r['specialty']}")

# ============== TAB 3: ENTITY EXTRACTION ==============
with tab3:
    st.subheader("Medical Entity Extraction")
    text_input_extract = st.text_area("Medical transcription:", height=200, key="extract_input")
    use_judge = st.checkbox("Also run evaluation (Groq judge)", value=True)

    if st.button("Extract"):
        if not text_input_extract.strip():
            st.warning("Please enter a transcription text.")
        else:
            from src.extractor import extract_medical_entities

            with st.spinner("Extracting (Ollama)..."):
                extracted = extract_medical_entities(text_input_extract)

            st.markdown("**Extracted entities:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Diagnoses:**", extracted.get("diagnoses", []))
                st.write("**Symptoms:**", extracted.get("symptoms", []))
            with col2:
                st.write("**Medications:**", extracted.get("medications", []))
                st.write("**Procedures:**", extracted.get("procedures", []))

            if use_judge:
                if not os.environ.get("GROQ_API_KEY"):
                    st.error("GROQ_API_KEY is not set in this terminal session.")
                else:
                    from src.extractor import evaluate_extraction_groq
                    with st.spinner("Evaluating (Groq)..."):
                        evaluation = evaluate_extraction_groq(text_input_extract, extracted)

                    if evaluation.get("overall_score") is not None:
                        st.markdown("**Evaluation (LLM-as-judge):**")
                        st.metric("Overall score", f"{evaluation['overall_score']:.2f}")

                        if evaluation.get("missing_items"):
                            st.write("**Missing items (present in text, not extracted):**")
                            st.json(evaluation["missing_items"])
                        if evaluation.get("hallucinations"):
                            st.write("**Hallucinations (extracted, not supported by text):**")
                            st.json(evaluation["hallucinations"])
                        if evaluation.get("explanation"):
                            st.info(evaluation["explanation"])
                    else:
                        st.warning("Evaluation did not return a valid result.")

            with st.expander("Source — original transcription"):
                st.write(text_input_extract)