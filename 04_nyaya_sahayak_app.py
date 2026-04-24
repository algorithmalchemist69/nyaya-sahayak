"""
Nyaya-Sahayak — Streamlit App
Run: streamlit run 04_nyaya_sahayak_app.py
"""

import os
import pickle

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ── Paths (local FAISS index built by 03_build_faiss_index.py) ───────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index", "bns.index")
META_PATH  = os.path.join(BASE_DIR, "faiss_index", "bns_metadata.pkl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Databricks Foundation Model API (open-source Llama 3.1 8B)
DATABRICKS_HOST  = "https://dbc-3f37515c-6bbb.cloud.databricks.com"
LLM_ENDPOINT     = "databricks-meta-llama-3-1-8b-instruct"

OFFENSE_HINTS = {
    "Theft":             "Chapter 17 — BNS Sections 303-305",
    "Robbery":           "Chapter 17 — BNS Sections 309-311",
    "Assault":           "Chapter 6  — BNS Sections 114-118",
    "Murder":            "Chapter 6  — BNS Sections 100-105",
    "Sexual Assault":    "Chapter 5  — BNS Sections 63-77",
    "Fraud":             "Chapter 17 — BNS Sections 316-318",
    "Cybercrime":        "BNS Section 316 + Information Technology Act",
    "Kidnapping":        "Chapter 6  — BNS Sections 137-144",
    "Domestic Violence": "Chapter 5  — BNS Sections 85-86",
    "Defamation":        "Chapter 19 — BNS Section 356",
    "Extortion":         "Chapter 17 — BNS Section 308",
    "Trespass":          "Chapter 17 — BNS Sections 329-332",
    "Rioting":           "Chapter 11 — BNS Sections 189-195",
    "Corruption":        "Chapter 12 — BNS Sections 61, 175-176",
    "Stalking":          "Chapter 5  — BNS Section 78",
    "Dowry Death":       "Chapter 5  — BNS Section 80",
}


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model …")
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading FAISS index …")
def load_faiss():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


@st.cache_resource
def get_llm_client(token: str):
    return OpenAI(
        api_key=token,
        base_url=f"{DATABRICKS_HOST}/serving-endpoints",
    )


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> list:
    model           = load_embed_model()
    index, metadata = load_faiss()
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(vec, top_k)
    out = []
    for sc, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            rec = metadata[idx].copy()
            rec["similarity"] = float(sc)
            out.append(rec)
    return out


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, client, max_tokens: int = 1024) -> str:
    response = client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Nyaya-Sahayak", page_icon="⚖️", layout="wide")
    st.title("⚖️ Nyaya-Sahayak")
    st.caption("Governance & Access to Justice — Spark · Delta Lake · FAISS · Llama 3.1 8B")

    with st.sidebar:
        st.header("Configuration")
        token = st.text_input(
            "Databricks Token",
            type="password",
            help="Your Databricks Personal Access Token (dapi...)",
        )
        st.divider()
        st.markdown(
            "**Stack**\n"
            "- Apache Spark + Delta Lake\n"
            "- FAISS (semantic retrieval)\n"
            "- Llama 3.1 8B (open-source)\n\n"
            "**Data**\n"
            "- Bharatiya Nyaya Sanhita, 2023\n"
            "- Synthetic FIR incident dataset"
        )

    if not token:
        st.warning("Enter your Databricks Token in the sidebar to start.")
        st.stop()

    client = get_llm_client(token)

    tab1, tab2 = st.tabs(["📖 Explain Like I'm 15", "🚨 FIR Category Helper"])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Legal Text Simplifier")
        st.write("Paste any BNS or Constitution text and get a plain-English explanation.")

        legal_input = st.text_area(
            "Legal text",
            placeholder="e.g. Whoever commits theft shall be punished with imprisonment...",
            height=180,
        )

        if st.button("Simplify ✨"):
            if not legal_input.strip():
                st.warning("Please paste some legal text first.")
            else:
                with st.spinner("Llama 3.1 is thinking …"):
                    result = call_llm(
                        system_prompt=(
                            "You are a friendly legal expert explaining Indian law "
                            "to a 15-year-old. Use short sentences, everyday words, "
                            "and relatable analogies. End with a one-line "
                            "'What this means for you' note."
                        ),
                        user_prompt=f"Explain simply (like I'm 15):\n\n{legal_input.strip()}",
                        client=client,
                    )
                st.success("Simple explanation:")
                st.markdown(result)

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("FIR Category Helper")
        st.write("Describe the incident — the app finds relevant BNS sections and guides you.")

        examples = [
            "Someone stole my bike",
            "My phone was hacked and money was taken from my bank",
            "Someone hit me with a stick during an argument",
            "My husband beats me and demands more dowry",
            "A contractor took advance payment and disappeared",
            "Local goons are demanding weekly protection money",
        ]

        choice = st.selectbox("Try an example", ["(type your own)"] + examples)
        incident_input = st.text_area(
            "Describe the incident",
            value="" if choice == "(type your own)" else choice,
            height=110,
        )
        top_k = st.slider("BNS sections to retrieve", 3, 10, 5)

        if st.button("Analyse & Suggest FIR Sections 🔍"):
            if not incident_input.strip():
                st.warning("Please describe the incident.")
            else:
                with st.spinner("Retrieving BNS sections via FAISS …"):
                    sections = retrieve(incident_input.strip(), top_k=top_k)

                with st.expander(f"Retrieved BNS Sections (top {top_k})", expanded=False):
                    for s in sections:
                        st.markdown(
                            f"**Section {s['Section']} — {s['Section_name']}** "
                            f"*(cosine: {s['similarity']:.3f})*\n\n"
                            f"{s['Description'][:400]}…"
                        )

                sections_text = "\n".join(
                    f"- Section {r['Section']} ({r['Section_name']}): {r['Description'][:300]}…"
                    for r in sections
                )
                offense_list = "\n".join(f"- {k}: {v}" for k, v in OFFENSE_HINTS.items())

                with st.spinner("Llama 3.1 is generating FIR guidance …"):
                    guidance = call_llm(
                        system_prompt=(
                            "You are a legal assistant helping Indian citizens file an FIR. "
                            "Identify the offense type, list relevant BNS sections, explain "
                            "why each applies in plain language, and advise what to do next. "
                            "Be empathetic and clear."
                        ),
                        user_prompt=(
                            f"Incident: {incident_input.strip()}\n\n"
                            f"Retrieved BNS sections:\n{sections_text}\n\n"
                            f"Offense type reference:\n{offense_list}"
                        ),
                        client=client,
                        max_tokens=1200,
                    )

                st.success("FIR Guidance")
                st.markdown(guidance)
                st.info(
                    "⚠️ AI-generated guidance for informational purposes only. "
                    "Visit your nearest police station or consult a lawyer to file an official FIR."
                )


if __name__ == "__main__":
    main()
