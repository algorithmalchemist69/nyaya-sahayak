# Databricks notebook source
# MAGIC %md
# MAGIC # Nyaya-Sahayak — Notebook 0: Setup
# MAGIC
# MAGIC Run this once per cluster restart.
# MAGIC
# MAGIC **Before running:**
# MAGIC 1. Upload `bns_sections.csv` and `synthetic_incidents.csv` to a Unity Catalog Volume or DBFS.
# MAGIC 2. Store your Anthropic API key in Databricks Secrets:
# MAGIC    ```
# MAGIC    databricks secrets create-scope nyaya
# MAGIC    databricks secrets put-secret nyaya anthropic_api_key --string-value sk-ant-...
# MAGIC    ```

# COMMAND ----------

# Install required libraries on the cluster
# (Alternatively add these in Cluster → Libraries UI)
# mlflow is pre-installed on Databricks — only need these two
%pip install faiss-cpu sentence-transformers --quiet

# COMMAND ----------

# Restart Python so the new packages are importable
dbutils.library.restartPython()

# COMMAND ----------

# ── Configuration — edit these to match your environment ──────────────────────

CATALOG   = "workspace"                  # Unity Catalog catalog name
SCHEMA    = "nyaya_sahayak"         # Schema / database name
VOLUME    = "raw_data"              # Volume for raw CSV uploads

# Derived paths
VOLUME_PATH   = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
BNS_CSV       = f"{VOLUME_PATH}/bns_sections.csv"
INCIDENTS_CSV = f"{VOLUME_PATH}/synthetic_incidents.csv"

# FAISS index stored on DBFS (driver-accessible)
FAISS_DIR  = "dbfs:/FileStore/nyaya_sahayak/faiss"
INDEX_PATH = f"{FAISS_DIR}/bns.index"
META_PATH  = f"{FAISS_DIR}/bns_metadata.pkl"

# Delta table full names
BNS_TABLE       = f"{CATALOG}.{SCHEMA}.bns_sections"
INCIDENTS_TABLE = f"{CATALOG}.{SCHEMA}.incidents"

# Anthropic API key (from Databricks Secrets)
# ANTHROPIC_API_KEY = dbutils.secrets.get("nyaya", "anthropic_api_key")

# ── Create catalog objects if they don't exist ────────────────────────────────
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

print(f"Catalog : {CATALOG}")
print(f"Schema  : {CATALOG}.{SCHEMA}")
print(f"Volume  : {VOLUME_PATH}")
print(f"FAISS   : {FAISS_DIR}")
print("\nSetup complete. Upload your CSV files to the Volume path shown above,")
print("then run Notebook 1 (ETL).")
