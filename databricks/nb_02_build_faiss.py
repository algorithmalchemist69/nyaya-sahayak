# Databricks notebook source
# MAGIC %md
# MAGIC # Nyaya-Sahayak — Notebook 2: Build FAISS Index
# MAGIC
# MAGIC **Flow:**
# MAGIC ```
# MAGIC Delta table (bns_sections)
# MAGIC     ──► collect to driver
# MAGIC     ──► SentenceTransformer embeddings (all-MiniLM-L6-v2)
# MAGIC     ──► FAISS IndexFlatIP (cosine similarity)
# MAGIC     ──► save index + metadata to DBFS
# MAGIC ```
# MAGIC
# MAGIC FAISS runs on the **driver node** — we collect BNS rows to the driver,
# MAGIC build the index in memory, then persist it to DBFS so the Streamlit app
# MAGIC can load it without re-building every time.

# COMMAND ----------

# ── Config ────────────────────────────────────────────────────────────────────

CATALOG = "workspace"
SCHEMA  = "nyaya_sahayak"

BNS_TABLE  = f"{CATALOG}.{SCHEMA}.bns_sections"
FAISS_DIR  = "/Volumes/workspace/nyaya_sahayak/raw_data/faiss"   # local path on driver
INDEX_PATH = f"{FAISS_DIR}/bns.index"
META_PATH  = f"{FAISS_DIR}/bns_metadata.pkl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# COMMAND ----------

# MAGIC %md ## 1. Load BNS sections from Delta table

# COMMAND ----------

import os, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

os.makedirs(FAISS_DIR, exist_ok=True)

# Read from Delta, collect to driver (358 rows — easily fits in RAM)
rows = spark.table(BNS_TABLE).collect()
records = [row.asDict() for row in rows]

print(f"Loaded {len(records)} BNS sections from Delta table '{BNS_TABLE}'")
print("Sample keys:", list(records[0].keys()))

# COMMAND ----------

# MAGIC %md ## 2. Generate embeddings

# COMMAND ----------

def build_text(record: dict) -> str:
    """Combine section name + description for richer semantic signal."""
    return f"{record.get('Section_name', '')}. {record.get('Description', '')}"

texts = [build_text(r) for r in records]

print(f"Loading model: {EMBED_MODEL} …")
model = SentenceTransformer(EMBED_MODEL)

print("Generating embeddings …")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # L2-normalise → cosine via inner product
)
embeddings = np.array(embeddings, dtype=np.float32)
print(f"Embedding matrix shape: {embeddings.shape}")

# COMMAND ----------

# MAGIC %md ## 3. Build & persist FAISS index

# COMMAND ----------

dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # inner product on normalised vecs == cosine sim
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "wb") as f:
    pickle.dump(records, f)

print(f"FAISS index saved : {INDEX_PATH}")
print(f"Metadata saved    : {META_PATH}")
print(f"Vectors indexed   : {index.ntotal}")

# COMMAND ----------

# MAGIC %md ## 4. Smoke-test retrieval

# COMMAND ----------

test_queries = [
    "someone stole my bicycle",
    "my phone was hacked and bank money stolen",
    "my husband beats me and demands dowry",
    "armed men robbed our shop at gunpoint",
]

for q in test_queries:
    vec = model.encode([q], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(vec, 3)
    print(f"\nQuery: '{q}'")
    for sc, idx in zip(scores[0], ids[0]):
        r = records[idx]
        print(f"  [{sc:.3f}]  Sec {r['Section']} — {r['Section_name']}")

# COMMAND ----------

print("\n✓ FAISS index built and validated.")
print(f"Run Notebook 3 (App) or deploy the Streamlit app via Databricks Apps.")
