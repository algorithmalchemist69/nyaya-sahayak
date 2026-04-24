# Databricks notebook source
# MAGIC %md
# MAGIC # Nyaya-Sahayak — Notebook 3: Inference Demo
# MAGIC
# MAGIC Interactive demo of both tasks inside a Databricks notebook.
# MAGIC
# MAGIC **Flow:**
# MAGIC ```
# MAGIC User input (widget, any Indian language)
# MAGIC     ──► Sarvam AI translate to English
# MAGIC     ──► FAISS retrieval (BNS sections)
# MAGIC     ──► Llama 3.1 8B via Databricks Foundation Model API
# MAGIC     ──► Sarvam AI translate back to user's language
# MAGIC     ──► displayHTML result
# MAGIC ```

# COMMAND ----------

%pip install faiss-cpu sentence-transformers "numpy<2.0" requests --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── Config ────────────────────────────────────────────────────────────────────

CATALOG = "workspace"
SCHEMA  = "nyaya_sahayak"

BNS_TABLE  = f"{CATALOG}.{SCHEMA}.bns_sections"
INDEX_PATH = "/Volumes/workspace/nyaya_sahayak/raw_data/faiss/bns.index"
META_PATH  = "/Volumes/workspace/nyaya_sahayak/raw_data/faiss/bns_metadata.pkl"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"
SARVAM_URL   = "https://api.sarvam.ai/translate"

LANGUAGES = {
    "English":   "en-IN",
    "Hindi":     "hi-IN",
    "Tamil":     "ta-IN",
    "Telugu":    "te-IN",
    "Kannada":   "kn-IN",
    "Malayalam": "ml-IN",
    "Bengali":   "bn-IN",
    "Marathi":   "mr-IN",
    "Gujarati":  "gu-IN",
    "Punjabi":   "pa-IN",
    "Odia":      "or-IN",
}

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

# COMMAND ----------

import pickle, requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from mlflow.deployments import get_deploy_client

embed_model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

llm_client = get_deploy_client("databricks")
print("Resources loaded.")

# COMMAND ----------

# ── Helpers ───────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> list:
    vec = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(vec, top_k)
    results = []
    for sc, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        rec = meta[idx].copy()
        rec["similarity"] = float(sc)
        results.append(rec)
    return results


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    response = llm_client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        },
    )
    return response["choices"][0]["message"]["content"]


def _translate_chunk(chunk: str, src: str, tgt: str, key: str) -> str:
    r = requests.post(
        SARVAM_URL,
        headers={"api-subscription-key": key},
        json={
            "input": chunk,
            "source_language_code": src,
            "target_language_code": tgt,
            "speaker_gender": "Female",
            "mode": "formal",
            "model": "mayura:v1",
            "enable_preprocessing": True,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("translated_text", chunk)


def translate(text: str, src: str, tgt: str, key: str) -> str:
    if src == tgt or not text.strip() or not key:
        return text
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > 900:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n" + para) if current else para
    if current:
        chunks.append(current.strip())
    return "\n".join(_translate_chunk(c, src, tgt, key) for c in chunks if c)

# COMMAND ----------

# MAGIC %md ## Widgets — set your inputs here

# COMMAND ----------

dbutils.widgets.text("sarvam_key", "", "Sarvam AI Key")
dbutils.widgets.dropdown("language", "English", list(LANGUAGES.keys()), "Language")
dbutils.widgets.text(
    "legal_text",
    "Whoever commits theft shall be punished with imprisonment of either description "
    "for a term which may extend to three years, or with fine, or with both.",
    "BNS / Constitution text"
)
dbutils.widgets.text("incident", "Someone stole my bicycle from outside my house", "Describe the incident")
dbutils.widgets.dropdown("top_k", "5", ["3","5","7","10"], "BNS sections to retrieve")

# COMMAND ----------

# MAGIC %md ## Task 1 — BNS Explanation

# COMMAND ----------

sarvam_key  = dbutils.widgets.get("sarvam_key")
lang_name   = dbutils.widgets.get("language")
lang_code   = LANGUAGES[lang_name]
legal_text  = dbutils.widgets.get("legal_text")

simplified = call_llm(
    system_prompt=(
        "You are a friendly legal expert explaining Indian law (BNS/Constitution) "
        "to a 15-year-old. Use short sentences, everyday words, and relatable analogies. "
        "After the explanation, add a one-line 'What this means for you' note."
    ),
    user_prompt=f"Explain in very simple English (like I'm 15):\n\n{legal_text}",
)

if lang_code != "en-IN":
    simplified = translate(simplified, "en-IN", lang_code, sarvam_key)

displayHTML(f"""
<div style="font-family:sans-serif; max-width:700px; padding:20px;
            border-left:4px solid #1976d2; background:#f5f9ff; border-radius:6px;">
  <h3 style="color:#1976d2;">📖 BNS Explanation ({lang_name})</h3>
  <p style="font-size:13px;color:#555;">Original: <em>{legal_text[:120]}…</em></p>
  <hr/>
  <div style="white-space:pre-wrap; font-size:15px; line-height:1.6;">{simplified}</div>
</div>
""")

# COMMAND ----------

# MAGIC %md ## Task 2 — FIR Category Helper

# COMMAND ----------

incident  = dbutils.widgets.get("incident")
top_k     = int(dbutils.widgets.get("top_k"))

# Translate incident to English for FAISS + LLM
english_incident = translate(incident, lang_code, "en-IN", sarvam_key)

sections = retrieve(english_incident, top_k=top_k)

sections_text = "\n".join(
    f"- Section {r['Section']} ({r['Section_name']}): {r['Description'][:300]}…"
    for r in sections
)
offense_list = "\n".join(f"- {k}: {v}" for k, v in OFFENSE_HINTS.items())

guidance = call_llm(
    system_prompt=(
        "You are a legal assistant helping Indian citizens file an FIR. "
        "Identify the offense type, list relevant BNS sections, explain why each applies "
        "in plain language, and advise what to do next. Be empathetic and clear."
    ),
    user_prompt=(
        f"Incident: {english_incident}\n\n"
        f"Retrieved BNS sections:\n{sections_text}\n\n"
        f"Offense type reference:\n{offense_list}"
    ),
    max_tokens=1200,
)

if lang_code != "en-IN":
    guidance = translate(guidance, "en-IN", lang_code, sarvam_key)

retrieved_html = "".join(
    f"<tr><td style='padding:6px;border-bottom:1px solid #eee;'><b>Sec {r['Section']}</b></td>"
    f"<td style='padding:6px;border-bottom:1px solid #eee;'>{r['Section_name']}</td>"
    f"<td style='padding:6px;border-bottom:1px solid #eee;text-align:right;color:#888;'>{r['similarity']:.3f}</td></tr>"
    for r in sections
)

displayHTML(f"""
<div style="font-family:sans-serif; max-width:800px;">
  <div style="background:#fff3e0; padding:16px; border-left:4px solid #f57c00; border-radius:6px; margin-bottom:16px;">
    <b>Incident ({lang_name}):</b> {incident}<br/>
    <small style="color:#888;">Translated to English: {english_incident}</small>
  </div>

  <details style="margin-bottom:16px;">
    <summary style="cursor:pointer; font-weight:bold; color:#1976d2;">
      Retrieved BNS Sections (FAISS — top {top_k})
    </summary>
    <table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:13px;">
      <thead><tr style="background:#f5f5f5;">
        <th style="padding:6px; text-align:left;">Section</th>
        <th style="padding:6px; text-align:left;">Name</th>
        <th style="padding:6px; text-align:right;">Similarity</th>
      </tr></thead>
      <tbody>{retrieved_html}</tbody>
    </table>
  </details>

  <div style="background:#e8f5e9; padding:20px; border-left:4px solid #388e3c; border-radius:6px;">
    <h3 style="color:#388e3c; margin-top:0;">🚨 FIR Guidance ({lang_name})</h3>
    <div style="white-space:pre-wrap; font-size:15px; line-height:1.7;">{guidance}</div>
    <hr style="border:none; border-top:1px solid #c8e6c9; margin:16px 0;"/>
    <small style="color:#666;">
      ⚠️ AI-generated guidance for informational purposes only.
      Visit your nearest police station or consult a lawyer to file an official FIR.
    </small>
  </div>
</div>
""")
