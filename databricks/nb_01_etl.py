# Databricks notebook source
# MAGIC %md
# MAGIC # Nyaya-Sahayak — Notebook 1: ETL → Delta Lake
# MAGIC
# MAGIC **Flow:**
# MAGIC ```
# MAGIC CSV (Volume) ──► Spark read ──► clean / transform ──► Delta tables (Unity Catalog)
# MAGIC ```
# MAGIC
# MAGIC Produces:
# MAGIC - `main.nyaya_sahayak.bns_sections`  — cleaned BNS law text
# MAGIC - `main.nyaya_sahayak.incidents`     — synthetic FIR incident dataset

# COMMAND ----------

# ── Config (keep in sync with nb_00_setup.py) ─────────────────────────────────

CATALOG   = "workspace"
SCHEMA    = "nyaya_sahayak"
VOLUME    = "raw_data"

VOLUME_PATH   = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
BNS_CSV       = f"{VOLUME_PATH}/bns_sections.csv"
INCIDENTS_CSV = f"{VOLUME_PATH}/synthetic_incidents.csv"

BNS_TABLE       = f"{CATALOG}.{SCHEMA}.bns_sections"
INCIDENTS_TABLE = f"{CATALOG}.{SCHEMA}.incidents"

# COMMAND ----------

# MAGIC %md ## 1. Ingest BNS Sections

# COMMAND ----------

import re
from pyspark.sql.functions import col, trim, length, regexp_replace

# Read raw CSV (multi-line descriptions need multiLine=true)
raw_bns = (
    spark.read
    .option("header", "true")
    .option("multiLine", "true")
    .option("quote", '"')
    .option("escape", '"')
    .option("encoding", "UTF-8")
    .csv(BNS_CSV)
)

# Normalise column names: collapse whitespace/underscore runs → single '_'
clean_cols = [re.sub(r'[\s_]+', '_', c).strip('_') for c in raw_bns.columns]
bns_df = raw_bns.toDF(*clean_cols)

print("Raw schema:")
bns_df.printSchema()

# COMMAND ----------

# Clean and filter
bns_df = (
    bns_df
    .withColumn("Chapter",       trim(col("Chapter")))
    .withColumn("Chapter_name",  trim(col("Chapter_name")))
    .withColumn("Chapter_subtype", trim(col("Chapter_subtype")))
    .withColumn("Section",       trim(col("Section")))
    .withColumn("Section_name",  trim(col("Section_name")))
    .withColumn("Description",   trim(regexp_replace(col("Description"), r'\r\n|\r|\n', ' ')))
    # Drop rows with trivially short descriptions
    .filter(length(col("Description")) > 20)
    # Drop header-like rows that got parsed as data
    .filter(col("Section") != "Section")
)

print(f"BNS sections after cleaning: {bns_df.count()}")
bns_df.show(5, truncate=80)

# COMMAND ----------

# Write to managed Delta table (Unity Catalog)
(
    bns_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BNS_TABLE)
)

print(f"✓ Written to Delta table: {BNS_TABLE}")

# COMMAND ----------

# Quick validation
spark.sql(f"""
    SELECT Chapter_name, COUNT(*) AS sections
    FROM {BNS_TABLE}
    GROUP BY Chapter_name
    ORDER BY Chapter_name
""").show(30, truncate=False)

# COMMAND ----------

# MAGIC %md ## 2. Ingest Synthetic Incident Dataset

# COMMAND ----------

incidents_df = (
    spark.read
    .option("header", "true")
    .option("encoding", "UTF-8")
    .csv(INCIDENTS_CSV)
    .withColumn("incident_description", trim(col("incident_description")))
    .withColumn("offense_type",         trim(col("offense_type")))
    .filter(length(col("incident_description")) > 5)
)

print(f"Incident records: {incidents_df.count()}")
incidents_df.show(10, truncate=80)

# COMMAND ----------

(
    incidents_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(INCIDENTS_TABLE)
)

print(f"✓ Written to Delta table: {INCIDENTS_TABLE}")

# COMMAND ----------

# Offense distribution
spark.sql(f"""
    SELECT offense_type, COUNT(*) AS count
    FROM {INCIDENTS_TABLE}
    GROUP BY offense_type
    ORDER BY count DESC
""").show(20, truncate=False)

# COMMAND ----------

# MAGIC %md ## 3. Enable Delta Lake Optimizations

# COMMAND ----------

# Auto-optimize future writes
spark.sql(f"ALTER TABLE {BNS_TABLE}       SET TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'delta.autoOptimize.autoCompact' = 'true')")
spark.sql(f"ALTER TABLE {INCIDENTS_TABLE} SET TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'delta.autoOptimize.autoCompact' = 'true')")

# Compute statistics for the query optimizer
spark.sql(f"ANALYZE TABLE {BNS_TABLE}       COMPUTE STATISTICS FOR ALL COLUMNS")
spark.sql(f"ANALYZE TABLE {INCIDENTS_TABLE} COMPUTE STATISTICS FOR ALL COLUMNS")

print("✓ Delta optimizations applied")
print("\nETL complete. Run Notebook 2 to build the FAISS index.")
