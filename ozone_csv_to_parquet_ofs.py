"""
=============================================================================
  Oracle CSV Extract → Clean → Parquet  |  Apache Ozone OFS + PySpark
=============================================================================
  Connection  : Native OFS  (ofs://<service-id>/<volume>/<bucket>/<path>)
  Cleaning    : Null handling, Deduplication, Column standardisation,
                Type casting, Invalid-row filtering (dates, numerics)
  Output      : Parquet partitioned by date/column on Ozone OFS
=============================================================================
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, IntegerType, LongType, DoubleType,
    DateType, TimestampType, DecimalType
)
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# =============================================================================
#  CONFIG  —  Edit this section to match your environment & Oracle schema
# =============================================================================

# ── Ozone OFS connection ─────────────────────────────────────────────────────
OFS_SERVICE_ID  = "om"                         # Ozone Manager service ID
OFS_OM_ADDRESS  = "<ozone-om-host>:9862"       # OM RPC address
OFS_OM_HA_NODES = ""                           # Comma-separated OM nodes (HA clusters)
                                               # e.g. "om1:9862,om2:9862,om3:9862"
                                               # Leave empty for non-HA

# ── Paths (ofs://<service-id>/<volume>/<bucket>/<path>/) ─────────────────────
INPUT_PATH  = "ofs://om/my-volume/oracle-extracts/csv/"
OUTPUT_PATH = "ofs://om/my-volume/oracle-extracts/parquet/"

# ── CSV options (tuned for Oracle extracts) ───────────────────────────────────
CSV_OPTIONS = {
    "header":                       "true",
    "inferSchema":                  "false",   # Explicit cast map below is safer
    "multiLine":                    "true",    # Handles Oracle CLOB embedded newlines
    "escape":                       '"',
    "quote":                        '"',
    "sep":                          ",",
    "encoding":                     "UTF-8",
    "dateFormat":                   "yyyy-MM-dd",
    "timestampFormat":              "yyyy-MM-dd HH:mm:ss",
    "nullValue":                    "",        # Oracle exports NULLs as empty strings
    "emptyValue":                   "",
    "ignoreLeadingWhiteSpace":      "true",
    "ignoreTrailingWhiteSpace":     "true",
}

# ── Parquet output ────────────────────────────────────────────────────────────
# Partition columns must exist in the DataFrame after cleaning.
# Common choices: ["extract_date"], ["region", "extract_date"]
PARQUET_PARTITION_BY = ["extract_date"]        # <-- set your partition column(s)
PARQUET_COMPRESSION  = "snappy"                # "snappy" | "gzip" | "zstd"
PARQUET_MODE         = "overwrite"             # "overwrite" | "append"

# ── Column type cast map ──────────────────────────────────────────────────────
# Keys are the RAW Oracle column names (ALL_CAPS). The script lower-cases them
# before casting, so the keys here must match the Oracle export headers exactly.
COLUMN_CAST_MAP = {
    "ID":             LongType(),
    "AMOUNT":         DoubleType(),
    "QUANTITY":       IntegerType(),
    "PRICE":          DecimalType(18, 4),
    "CREATED_DATE":   DateType(),
    "UPDATED_TS":     TimestampType(),
    "EXTRACT_DATE":   DateType(),              # Used as partition column
}

# ── Null / missing-value rules ────────────────────────────────────────────────
# Rows with NULL in any of these columns are DROPPED entirely.
MANDATORY_COLS = ["ID", "CREATED_DATE"]       # Maps to lower-cased names

# For non-mandatory columns, fill NULLs with a default value instead of dropping.
# Format: { "col_name_lower": fill_value }
NULL_FILL_MAP = {
    "status":      "UNKNOWN",
    "region":      "N/A",
    "quantity":    0,
    "amount":      0.0,
}

# ── Invalid-row filter rules ──────────────────────────────────────────────────
# Rows where these numeric columns are NOT strictly > 0 are removed.
POSITIVE_NUMERIC_COLS = ["amount", "quantity"]   # lower-case

# Rows where CREATED_DATE falls outside this range are removed.
DATE_COL  = "created_date"                       # lower-case
DATE_MIN  = "2000-01-01"
DATE_MAX  = "2099-12-31"

# Columns to drop before writing (Oracle row metadata etc.)
DROP_COLS = ["rowid", "export_ts", "ora_rowscn"]


# =============================================================================
#  SPARK SESSION
# =============================================================================

def build_spark_session() -> SparkSession:
    log.info("Building SparkSession with OFS configuration …")
    builder = (
        SparkSession.builder
        .appName("OracleCSV_OFS_to_Parquet")
        # ── OFS filesystem ────────────────────────────────────────────────
        .config("spark.hadoop.fs.ofs.impl",
                "org.apache.hadoop.fs.ozone.OzoneFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.ofs.impl",
                "org.apache.hadoop.fs.ozone.OzFs")
    )

    if OFS_OM_HA_NODES:
        # HA Ozone Manager setup
        om_nodes = [n.strip() for n in OFS_OM_HA_NODES.split(",")]
        node_ids  = ",".join([f"om{i+1}" for i in range(len(om_nodes))])
        builder = builder.config(
            f"spark.hadoop.ozone.om.service.ids", OFS_SERVICE_ID
        ).config(
            f"spark.hadoop.ozone.om.nodes.{OFS_SERVICE_ID}", node_ids
        )
        for i, addr in enumerate(om_nodes):
            builder = builder.config(
                f"spark.hadoop.ozone.om.address.{OFS_SERVICE_ID}.om{i+1}", addr
            )
    else:
        # Single Ozone Manager
        builder = builder.config(
            f"spark.hadoop.ozone.om.address.{OFS_SERVICE_ID}", OFS_OM_ADDRESS
        )

    builder = (
        builder
        # ── Parquet ───────────────────────────────────────────────────────
        .config("spark.sql.parquet.compression.codec",     PARQUET_COMPRESSION)
        .config("spark.sql.parquet.mergeSchema",           "false")
        .config("spark.sql.files.maxRecordsPerFile",       "500000")
        # ── Performance ───────────────────────────────────────────────────
        .config("spark.sql.adaptive.enabled",              "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions",            "200")
        # ── Stability ─────────────────────────────────────────────────────
        .config("spark.sql.legacy.timeParserPolicy",       "LEGACY")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info("SparkSession ready.")
    return spark


# =============================================================================
#  EXTRACT
# =============================================================================

def extract(spark: SparkSession) -> DataFrame:
    log.info(f"[EXTRACT] Reading CSV files from OFS: {INPUT_PATH}")
    df = spark.read.options(**CSV_OPTIONS).csv(INPUT_PATH)
    log.info(f"[EXTRACT] Raw schema:")
    df.printSchema()
    log.info(f"[EXTRACT] Raw row count: {df.count():,}")
    return df


# =============================================================================
#  TRANSFORM  —  Cleaning pipeline
# =============================================================================

def transform(df: DataFrame) -> DataFrame:
    log.info("[TRANSFORM] Starting cleaning pipeline …")
    df = _standardize_column_names(df)
    df = _cast_columns(df)
    df = _drop_unwanted_columns(df)
    df = _trim_and_nullify_strings(df)
    df = _fill_nulls(df)
    df = _drop_mandatory_nulls(df)
    df = _remove_duplicates(df)
    df = _filter_invalid_rows(df)
    log.info(f"[TRANSFORM] Clean row count: {df.count():,}")
    df.printSchema()
    return df


def _standardize_column_names(df: DataFrame) -> DataFrame:
    """
    Convert Oracle ALL_CAPS column headers to lowercase snake_case.
    e.g.  CREATED DATE  →  created_date
          CUSTOMER-ID   →  customer_id
    """
    for original in df.columns:
        normalised = (
            original.strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace(".", "_")
        )
        if original != normalised:
            df = df.withColumnRenamed(original, normalised)
    log.info("[TRANSFORM] ✓ Column names standardised to snake_case.")
    return df


def _cast_columns(df: DataFrame) -> DataFrame:
    """Apply explicit type casts from COLUMN_CAST_MAP."""
    for raw_col, dtype in COLUMN_CAST_MAP.items():
        col_lower = raw_col.lower()
        if col_lower in df.columns:
            df = df.withColumn(col_lower, F.col(col_lower).cast(dtype))
            log.info(f"[TRANSFORM]   Cast  {col_lower}  →  {dtype}")
    log.info("[TRANSFORM] ✓ Column types applied.")
    return df


def _drop_unwanted_columns(df: DataFrame) -> DataFrame:
    """Remove Oracle internal / export metadata columns if present."""
    cols_present = [c for c in DROP_COLS if c in df.columns]
    if cols_present:
        df = df.drop(*cols_present)
        log.info(f"[TRANSFORM] ✓ Dropped columns: {cols_present}")
    return df


def _trim_and_nullify_strings(df: DataFrame) -> DataFrame:
    """
    For every StringType column:
      1. Strip leading / trailing whitespace.
      2. Replace empty string ('') with NULL — Oracle often exports '' for NULL.
    """
    string_cols = [f.name for f in df.schema.fields
                   if isinstance(f.dataType, StringType)]
    for c in string_cols:
        df = df.withColumn(
            c,
            F.when(F.trim(F.col(c)) == "", None)
             .otherwise(F.trim(F.col(c)))
        )
    log.info(f"[TRANSFORM] ✓ Trimmed & null-ified {len(string_cols)} string column(s).")
    return df


def _fill_nulls(df: DataFrame) -> DataFrame:
    """Fill NULLs in non-mandatory columns using NULL_FILL_MAP."""
    applicable = {k: v for k, v in NULL_FILL_MAP.items() if k in df.columns}
    if applicable:
        df = df.fillna(applicable)
        log.info(f"[TRANSFORM] ✓ Filled NULLs in: {list(applicable.keys())}")
    return df


def _drop_mandatory_nulls(df: DataFrame) -> DataFrame:
    """Drop rows that have NULL in any MANDATORY_COLS column."""
    cols = [c.lower() for c in MANDATORY_COLS if c.lower() in df.columns]
    if not cols:
        return df
    before = df.count()
    df = df.dropna(subset=cols)
    removed = before - df.count()
    log.info(f"[TRANSFORM] ✓ Dropped {removed:,} rows with NULLs in {cols}.")
    return df


def _remove_duplicates(df: DataFrame) -> DataFrame:
    """Full-row deduplication."""
    before = df.count()
    df = df.dropDuplicates()
    removed = before - df.count()
    log.info(f"[TRANSFORM] ✓ Removed {removed:,} duplicate rows.")
    return df


def _filter_invalid_rows(df: DataFrame) -> DataFrame:
    """
    Remove rows with:
      • Non-positive values in POSITIVE_NUMERIC_COLS
      • DATE_COL outside [DATE_MIN, DATE_MAX]
    """
    # Positive numeric filter
    for c in POSITIVE_NUMERIC_COLS:
        if c in df.columns:
            before = df.count()
            df = df.filter(F.col(c) > 0)
            log.info(f"[TRANSFORM] ✓ Removed {before - df.count():,} rows where {c} ≤ 0.")

    # Date range filter
    if DATE_COL in df.columns:
        before = df.count()
        df = df.filter(F.col(DATE_COL).between(DATE_MIN, DATE_MAX))
        log.info(
            f"[TRANSFORM] ✓ Removed {before - df.count():,} rows where "
            f"{DATE_COL} ∉ [{DATE_MIN}, {DATE_MAX}]."
        )

    return df


# =============================================================================
#  LOAD  —  Write partitioned Parquet back to Ozone OFS
# =============================================================================

def load(df: DataFrame) -> None:
    log.info(f"[LOAD] Writing Parquet to OFS: {OUTPUT_PATH}")
    log.info(f"[LOAD] Partition columns : {PARQUET_PARTITION_BY}")
    log.info(f"[LOAD] Compression       : {PARQUET_COMPRESSION}")
    log.info(f"[LOAD] Write mode        : {PARQUET_MODE}")

    # Validate partition columns exist
    missing = [c for c in PARQUET_PARTITION_BY if c not in df.columns]
    if missing:
        raise ValueError(
            f"Partition column(s) not found in DataFrame: {missing}. "
            f"Available columns: {df.columns}"
        )

    (
        df.write
          .mode(PARQUET_MODE)
          .option("compression", PARQUET_COMPRESSION)
          .partitionBy(*PARQUET_PARTITION_BY)
          .parquet(OUTPUT_PATH)
    )
    log.info("[LOAD] ✓ Parquet write complete.")


# =============================================================================
#  DATA QUALITY REPORT  —  Quick summary before write
# =============================================================================

def print_quality_report(df: DataFrame) -> None:
    log.info("=" * 60)
    log.info("DATA QUALITY REPORT")
    log.info("=" * 60)
    log.info(f"Total rows   : {df.count():,}")
    log.info(f"Total columns: {len(df.columns)}")
    log.info("Null counts per column:")
    null_counts = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    for col_name, null_count in sorted(null_counts.items()):
        if null_count > 0:
            log.info(f"  {col_name:<30} {null_count:>10,} nulls")
    log.info("=" * 60)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    spark = build_spark_session()
    try:
        raw_df   = extract(spark)
        clean_df = transform(raw_df)
        print_quality_report(clean_df)
        load(clean_df)
        log.info("Pipeline completed successfully ✓")
    except Exception as exc:
        log.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
