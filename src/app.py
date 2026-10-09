import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Agriculture QA", page_icon="🌱", layout="wide")

MODEL_ID = "Toronga/flan-t5-agri-qa"   # HuggingFace, NOT GitHub

@st.cache_resource
def load_models():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    mdl.eval()
    mdl.generation_config.max_length = None
    mdl.generation_config.max_new_tokens = 128
    mdl.generation_config.min_length = 5
    mdl.generation_config.num_beams = 4
    mdl.generation_config.no_repeat_ngram_size = 3
    return tok, mdl

@st.cache_resource
def load_tfidf():
    df = pd.read_csv("data/train_data.csv")
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
    mat = vec.fit_transform(df["question"])
    return df, vec, mat

tok, mdl = load_models()
train_df, vectorizer, tfidf_matrix = load_tfidf()

def ft_answer(q):
    inp = tok("question: " + q, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        out = mdl.generate(**inp)
    return tok.decode(out[0], skip_special_tokens=True)

def tfidf_answer(q):
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix)
    idx = sims.argmax()
    return train_df.iloc[idx]["answer"], float(sims[0, idx])

st.title("🌱 Agriculture Domain Question Answering")
st.markdown(
    "Fine-tuned **flan-t5-base** on 20k agricultural Q&A pairs, "
    "benchmarked against a **TF-IDF retrieval baseline**."
)

mode = st.radio(
    "Answering mode:",
    ["Fine-tuned T5", "TF-IDF Retrieval", "Both (compare)"],
    index=2, horizontal=True,
)

question = st.text_area(
    "Ask your agricultural question:",
    height=80,
    placeholder="e.g. how to control aphids in sugarcane",
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown("### Results")
        if mode in ["Fine-tuned T5", "Both (compare)"]:
            st.markdown("**Fine-tuned T5 (generative):**")
            st.info(ft_answer(question))
        if mode in ["TF-IDF Retrieval", "Both (compare)"]:
            ans, score = tfidf_answer(question)
            st.markdown(f"**TF-IDF Retrieval (similarity = {score:.3f}):**")
            st.success(ans)

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Model:** google/flan-t5-base, 3 epochs\n"
        "- **Train:** 8,000 agricultural Q&A pairs\n"
        "- **Val ROUGE-L:** 0.316\n"
        "- **BERTScore F1:** 0.878"
    )
