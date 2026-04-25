# Databricks notebook source
# MAGIC %md
# MAGIC # Nyaya-Sahayak — Notebook 3: Inference Demo
# MAGIC
# MAGIC Interactive demo of both tasks inside a Databricks notebook.
# MAGIC
# MAGIC **Flow:**
# MAGIC ```
# MAGIC User input (widget / audio file in Volume)
# MAGIC     ──► Sarvam AI STT  (if audio file provided)
# MAGIC     ──► Sarvam AI translate to English
# MAGIC     ──► FAISS retrieval (BNS sections)
# MAGIC     ──► Llama 3.1 8B via Databricks Foundation Model API
# MAGIC     ──► Sarvam AI TTS  (voice output embedded in HTML)
# MAGIC     ──► displayHTML result
# MAGIC ```

# COMMAND ----------

%pip install faiss-cpu sentence-transformers "numpy<2.0" requests langdetect --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── Config ────────────────────────────────────────────────────────────────────

CATALOG = "workspace"
SCHEMA  = "nyaya_sahayak"

BNS_TABLE    = f"{CATALOG}.{SCHEMA}.bns_sections"
INDEX_PATH   = "/Volumes/workspace/nyaya_sahayak/raw_data/faiss/bns.index"
META_PATH    = "/Volumes/workspace/nyaya_sahayak/raw_data/faiss/bns_metadata.pkl"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"
SARVAM_URL   = "https://api.sarvam.ai/translate"
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

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

LANGDETECT_CODES = {
    "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
    "Malayalam": "ml", "Bengali": "bn", "Marathi": "mr",
    "Gujarati": "gu", "Punjabi": "pa", "Odia": "or",
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

import base64
import pickle
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from mlflow.deployments import get_deploy_client
from langdetect import detect, LangDetectException

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


def translate_text(text: str, src: str, tgt: str, key: str) -> str:
    """Translate text via Sarvam AI, chunking into ≤900-char pieces."""
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

    translated = []
    for chunk in chunks:
        if not chunk:
            continue
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
        translated.append(r.json().get("translated_text", chunk))
    return "\n".join(translated)


def translate_input(text: str, src: str, key: str) -> str:
    return translate_text(text, src, "en-IN", key)


def speech_to_text(file_path: str, lang_code: str, key: str) -> str:
    """Transcribe an audio file from a Volume path using Sarvam AI STT."""
    if not file_path.strip() or not key:
        return ""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        fname = file_path.split("/")[-1]
        mime  = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/webm"
        r = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": key},
            files={"file": (fname, audio_bytes, mime)},
            data={"model": "saarika:v2", "language_code": lang_code},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("transcript", "")
    except Exception as e:
        print(f"STT error: {e}")
        return ""


def text_to_speech(text: str, lang_code: str, key: str) -> str:
    """Convert text to speech via Sarvam AI. Returns base64 WAV string for HTML embedding."""
    if not key or lang_code == "en-IN" or not text.strip():
        return ""
    try:
        r = requests.post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            json={
                "inputs":               [text[:500]],
                "target_language_code": lang_code,
                "speaker":              "meera",
                "pitch":                0,
                "pace":                 1.0,
                "loudness":             1.5,
                "speech_sample_rate":   8000,
                "enable_preprocessing": True,
                "model":                "bulbul:v1",
            },
            timeout=60,
        )
        r.raise_for_status()
        audios = r.json().get("audios", [])
        return audios[0] if audios else ""
    except Exception as e:
        print(f"TTS error: {e}")
        return ""


def bhasha_bench_score(text: str, lang_name: str) -> dict:
    """Compute BhashaBench metrics for a non-English LLM output."""
    expected = LANGDETECT_CODES.get(lang_name, "")
    try:
        detected = detect(text)
        lang_score = 1.0 if detected == expected else 0.0
    except LangDetectException:
        detected = "unknown"
        lang_score = 0.0
    composite = round(lang_score * 100)
    return {
        "language_detection": composite,
        "composite":          composite,
        "detected_lang":      detected,
        "expected_lang":      expected,
    }

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
dbutils.widgets.text("stt_path", "", "Audio file path in Volume (optional, for voice input)")

# COMMAND ----------

# MAGIC %md ## Task 1 — BNS Explanation

# COMMAND ----------

sarvam_key  = dbutils.widgets.get("sarvam_key")
lang_name   = dbutils.widgets.get("language")
lang_code   = LANGUAGES[lang_name]
legal_text  = dbutils.widgets.get("legal_text")
stt_path    = dbutils.widgets.get("stt_path")

lang_instruction = f"Respond in {lang_name}." if lang_name != "English" else ""

# Voice input for Task 1 — if an audio file path is provided, transcribe it
if stt_path.strip() and sarvam_key:
    transcribed = speech_to_text(stt_path.strip(), lang_code, sarvam_key)
    if transcribed:
        print(f"STT transcription: {transcribed}")
        legal_text = transcribed

simplified = call_llm(
    system_prompt=(
        f"You are a friendly legal expert explaining Indian law (BNS/Constitution) "
        f"to a 15-year-old. Use short sentences, everyday words, and relatable analogies. "
        f"End with a one-line 'What this means for you' note. {lang_instruction}"
    ),
    user_prompt=f"Explain simply:\n\n{legal_text}",
)

# Voice output
audio_b64_1 = ""
if lang_name != "English" and sarvam_key:
    audio_b64_1 = text_to_speech(simplified, lang_code, sarvam_key)

audio_html_1 = ""
if audio_b64_1:
    audio_html_1 = f"""
  <div style="margin-top:12px;">
    <p style="font-size:13px;color:#1565c0;margin:0 0 6px;">&#128266; Voice output (first ~500 characters)</p>
    <audio controls style="width:100%;">
      <source src="data:audio/wav;base64,{audio_b64_1}" type="audio/wav"/>
    </audio>
  </div>
"""

# BhashaBench score
bb1 = {}
if lang_name != "English":
    bb1 = bhasha_bench_score(simplified, lang_name)

bb1_html = ""
if bb1:
    bb1_html = f"""
  <hr style="border:none;border-top:1px solid #c5d8f8;margin:16px 0;"/>
  <h4 style="color:#1976d2;margin:0 0 10px;">&#128202; BhashaBench Score</h4>
  <div style="display:flex;gap:16px;">
    <div style="background:#e3f2fd;padding:12px 20px;border-radius:8px;text-align:center;flex:1;">
      <div style="font-size:22px;font-weight:bold;color:#1565c0;">{bb1['language_detection']} / 100</div>
      <div style="font-size:12px;color:#555;margin-top:4px;">Language Detection<br/><small>detected: {bb1['detected_lang']} &middot; expected: {bb1['expected_lang']}</small></div>
    </div>
    <div style="background:#1976d2;padding:12px 20px;border-radius:8px;text-align:center;flex:1;">
      <div style="font-size:22px;font-weight:bold;color:#fff;">{bb1['composite']} / 100</div>
      <div style="font-size:12px;color:#bbdefb;margin-top:4px;">BhashaBench Score<br/><small>language detection based</small></div>
    </div>
  </div>
"""

displayHTML(f"""
<div style="font-family:sans-serif; max-width:700px; padding:20px;
            border-left:4px solid #1976d2; background:#f5f9ff; border-radius:6px;">
  <h3 style="color:#1976d2;">&#128214; BNS Explanation ({lang_name})</h3>
  <p style="font-size:13px;color:#555;">Original: <em>{legal_text[:120]}&#8230;</em></p>
  <hr/>
  <div style="white-space:pre-wrap; font-size:15px; line-height:1.6;">{simplified}</div>
  {audio_html_1}
  {bb1_html}
</div>
""")

# COMMAND ----------

# MAGIC %md ## Task 2 — FIR Category Helper

# COMMAND ----------

incident = dbutils.widgets.get("incident")
top_k    = int(dbutils.widgets.get("top_k"))

# Voice input for Task 2 — if an audio file path is provided, transcribe it
if stt_path.strip() and sarvam_key:
    transcribed2 = speech_to_text(stt_path.strip(), lang_code, sarvam_key)
    if transcribed2:
        print(f"STT transcription: {transcribed2}")
        incident = transcribed2

# Translate user input to English for FAISS retrieval
english_incident = translate_input(incident, lang_code, sarvam_key)

sections = retrieve(english_incident, top_k=top_k)

sections_text = "\n".join(
    f"- Section {r['Section']} ({r['Section_name']}): {r['Description'][:300]}…"
    for r in sections
)
offense_list = "\n".join(f"- {k}: {v}" for k, v in OFFENSE_HINTS.items())

guidance = call_llm(
    system_prompt=(
        f"You are a legal assistant helping Indian citizens file an FIR. "
        f"Identify the offense type, list relevant BNS sections, explain why each applies "
        f"in plain language, and advise what to do next. Be empathetic and clear. "
        f"{lang_instruction}"
    ),
    user_prompt=(
        f"Incident: {english_incident}\n\n"
        f"Retrieved BNS sections:\n{sections_text}\n\n"
        f"Offense type reference:\n{offense_list}"
    ),
    max_tokens=1200,
)

# Voice output
audio_b64_2 = ""
if lang_name != "English" and sarvam_key:
    audio_b64_2 = text_to_speech(guidance, lang_code, sarvam_key)

audio_html_2 = ""
if audio_b64_2:
    audio_html_2 = f"""
  <div style="margin-top:12px;">
    <p style="font-size:13px;color:#2e7d32;margin:0 0 6px;">&#128266; Voice output (first ~500 characters)</p>
    <audio controls style="width:100%;">
      <source src="data:audio/wav;base64,{audio_b64_2}" type="audio/wav"/>
    </audio>
  </div>
"""

retrieved_html = "".join(
    f"<tr><td style='padding:6px;border-bottom:1px solid #eee;'><b>Sec {r['Section']}</b></td>"
    f"<td style='padding:6px;border-bottom:1px solid #eee;'>{r['Section_name']}</td>"
    f"<td style='padding:6px;border-bottom:1px solid #eee;text-align:right;color:#888;'>{r['similarity']:.3f}</td></tr>"
    for r in sections
)

# BhashaBench score
bb2 = {}
if lang_name != "English":
    bb2 = bhasha_bench_score(guidance, lang_name)

bb2_html = ""
if bb2:
    bb2_html = f"""
  <hr style="border:none;border-top:1px solid #c8e6c9;margin:16px 0;"/>
  <h4 style="color:#388e3c;margin:0 0 10px;">&#128202; BhashaBench Score</h4>
  <div style="display:flex;gap:16px;">
    <div style="background:#f1f8e9;padding:12px 20px;border-radius:8px;text-align:center;flex:1;">
      <div style="font-size:22px;font-weight:bold;color:#2e7d32;">{bb2['language_detection']} / 100</div>
      <div style="font-size:12px;color:#555;margin-top:4px;">Language Detection<br/><small>detected: {bb2['detected_lang']} &middot; expected: {bb2['expected_lang']}</small></div>
    </div>
    <div style="background:#388e3c;padding:12px 20px;border-radius:8px;text-align:center;flex:1;">
      <div style="font-size:22px;font-weight:bold;color:#fff;">{bb2['composite']} / 100</div>
      <div style="font-size:12px;color:#c8e6c9;margin-top:4px;">BhashaBench Score<br/><small>language detection based</small></div>
    </div>
  </div>
"""

displayHTML(f"""
<div style="font-family:sans-serif; max-width:800px;">
  <div style="background:#fff3e0; padding:16px; border-left:4px solid #f57c00; border-radius:6px; margin-bottom:16px;">
    <b>Incident ({lang_name}):</b> {incident}<br/>
    <small style="color:#888;">Translated to English: {english_incident}</small>
  </div>

  <details style="margin-bottom:16px;">
    <summary style="cursor:pointer; font-weight:bold; color:#1976d2;">
      Retrieved BNS Sections (FAISS &#8212; top {top_k})
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
    <h3 style="color:#388e3c; margin-top:0;">&#128680; FIR Guidance ({lang_name})</h3>
    <div style="white-space:pre-wrap; font-size:15px; line-height:1.7;">{guidance}</div>
    {audio_html_2}
    {bb2_html}
    <hr style="border:none; border-top:1px solid #c8e6c9; margin:16px 0;"/>
    <small style="color:#666;">
      &#9888;&#65039; AI-generated guidance for informational purposes only.
      Visit your nearest police station or consult a lawyer to file an official FIR.
    </small>
  </div>
</div>
""")
