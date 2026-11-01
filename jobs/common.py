"""Shared settings and helpers for the TasteTrends Spark jobs.

Paths come from environment variables so the same code runs on the Docker cluster
(data in HDFS) and on a single laptop (data in a local folder):

  TT_DATA     where the raw and cleaned data live   default hdfs://namenode:9000/tastetrends
  TT_RESULTS  local folder for result files         default /opt/results
              (written by the driver only, so executors never write to the bind mount)
"""
import json
import os
import socket
import time

from pyspark.sql import SparkSession, functions as F, types as T

DATA = os.environ.get("TT_DATA", "hdfs://namenode:9000/tastetrends")
RESULTS = os.environ.get("TT_RESULTS", "/opt/results")
HERE = os.path.dirname(os.path.abspath(__file__))
SQL_DIR = os.path.join(HERE, "sql")

# One cuisine per restaurant, first match wins (more specific cuisines first)
CUISINES = ["Indian", "Thai", "Japanese", "Chinese", "Mexican", "Italian",
            "Mediterranean", "Seafood", "Pizza", "Burgers", "Breakfast & Brunch",
            "Fast Food", "American"]

MIN_TRAIN = 5          # a test user needs at least this many training reviews
N_EVAL_USERS = 5000    # cap on test users

# "categories" is read as a string. In the 2013 edition it is a JSON list and Spark keeps the
# raw text (["Mexican", "Restaurants"]); in newer editions it is already a comma-separated
# string. A substring test works for both, so no conversion step is needed.
BUSINESS_SCHEMA = T.StructType([
    T.StructField("business_id", T.StringType()),
    T.StructField("name", T.StringType()),
    T.StructField("city", T.StringType()),
    T.StructField("state", T.StringType()),
    T.StructField("stars", T.DoubleType()),
    T.StructField("review_count", T.IntegerType()),
    T.StructField("categories", T.StringType()),
])

REVIEW_SCHEMA = T.StructType([
    T.StructField("review_id", T.StringType()),
    T.StructField("user_id", T.StringType()),
    T.StructField("business_id", T.StringType()),
    T.StructField("stars", T.DoubleType()),
    T.StructField("date", T.StringType()),
    T.StructField("text", T.StringType()),
])


def spark_session(name):
    return (SparkSession.builder.appName(name)
            .config("spark.sql.shuffle.partitions", os.environ.get("TT_SHUFFLE_PARTITIONS", "8"))
            # files Spark writes to HDFS get 2 copies, matching the 2 DataNodes
            .config("spark.hadoop.dfs.replication", "2")
            .getOrCreate())


def year_window(reviews):
    """Years used by the trend analyses and the recommender, taken from the data so the same code
    works on any Yelp edition. The last year is usually partial, so:
      last_full  = last year - 1   (2012 for the 2013 edition, 2021 for the 2022 edition)
      first      = last_full - 5   (six full years for the trend charts)
      cutoff     = last_full       (recommender trains before it, tests from it on; TT_CUTOFF overrides)
    """
    last = reviews.agg(F.max("year")).first()[0]
    last_full = last - 1
    cutoff = int(os.environ.get("TT_CUTOFF") or last_full)
    return last_full - 5, last_full, cutoff


def update_summary(**values):
    """Add values to results/dataset_summary.json (read by the dashboard through MongoDB)."""
    path = results_path("dataset_summary.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data.update(values)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cuisine_expr(col="categories"):
    expr = F.lit("Other")
    for c in reversed(CUISINES):
        expr = F.when(F.col(col).contains(c), F.lit(c)).otherwise(expr)
    return expr


def clean_text(col):
    """Lowercase, keep letters only, collapse whitespace."""
    t = F.lower(F.coalesce(col, F.lit("")))
    t = F.regexp_replace(t, "[^a-z ]", " ")
    t = F.regexp_replace(t, "\\s+", " ")
    return F.trim(t)


def results_path(*parts):
    p = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def save_csv(df, name):
    """Small result tables: collect to the driver and write one CSV."""
    pdf = df.toPandas()
    pdf.to_csv(results_path(name), index=False)
    return pdf


def cluster_info(spark):
    sc = spark.sparkContext
    executors = sc._jsc.sc().getExecutorMemoryStatus().size() - 1  # minus the driver
    return {"master": sc.master, "app_id": sc.applicationId,
            "executors": max(executors, 0) if not sc.master.startswith("local") else 0,
            "default_parallelism": sc.defaultParallelism,
            "driver_host": socket.gethostname()}


def log_run(job, started, spark, **stats):
    """Append one line per job to results/run_log.jsonl (shown on the dashboard)."""
    row = {"job": job, "seconds": round(time.time() - started, 1),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    row.update(cluster_info(spark))
    row.update(stats)
    with open(results_path("run_log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
