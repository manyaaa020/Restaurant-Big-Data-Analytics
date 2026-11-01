"""Step 1 - Ingest and clean (PySpark ETL).

Reads the raw Yelp JSON from HDFS, keeps restaurant reviews, assigns one cuisine per
restaurant, cleans the text and writes Parquet back to HDFS for the later steps.

docker exec spark-master spark-submit ... /opt/jobs/01_clean.py
"""
import json
import os
import socket
import time

from pyspark.sql import functions as F
from common import (DATA, BUSINESS_SCHEMA, REVIEW_SCHEMA, spark_session, cuisine_expr,
                    clean_text, save_csv, results_path, log_run)

started = time.time()
spark = spark_session("tastetrends-01-clean")

biz_raw = spark.read.schema(BUSINESS_SCHEMA).json(f"{DATA}/raw/business")
rev_raw = spark.read.schema(REVIEW_SCHEMA).json(f"{DATA}/raw/review")


# Evidence that the work is distributed: which worker processed each input partition
def where(i, rows):
    node = os.environ.get("TT_NODE_NAME", socket.gethostname())
    yield (i, node, sum(1 for _ in rows))


dist = rev_raw.rdd.mapPartitionsWithIndex(where).toDF(["partition", "worker", "rows"])
save_csv(dist.orderBy("partition"), "partition_distribution.csv")

# 1. Businesses -> restaurants with one cuisine
restaurants = (biz_raw
    .filter(F.col("business_id").isNotNull() & F.col("categories").contains("Restaurants"))
    .select("business_id", "name", "city", "state",
            F.col("stars").alias("biz_stars"), "review_count",
            cuisine_expr().alias("cuisine"))
    .cache())

# 2. Reviews -> restaurant reviews only. The restaurant table is small, so it is broadcast
#    to every executor and the large review table is never shuffled.
reviews = (rev_raw
    .filter(F.col("review_id").isNotNull() & F.col("business_id").isNotNull()
            & F.col("stars").isNotNull())
    .join(F.broadcast(restaurants.select("business_id", "name", "city", "cuisine")), "business_id")
    .select("review_id", "user_id", "business_id", "name", "city", "cuisine",
            F.col("stars").cast("int").alias("stars"),
            F.substring("date", 1, 4).cast("int").alias("year"),
            F.col("text"),
            clean_text(F.col("text")).alias("text_clean")))

restaurants.write.mode("overwrite").parquet(f"{DATA}/clean/restaurants")
reviews.write.mode("overwrite").parquet(f"{DATA}/clean/reviews")

# Counts for the report and dashboard (read back from the written Parquet)
clean = spark.read.parquet(f"{DATA}/clean/reviews")
stats = {
    "raw_businesses": biz_raw.count(),
    "raw_reviews": rev_raw.count(),
    "restaurants": restaurants.count(),
    "restaurant_reviews": clean.count(),
    "negative_reviews": clean.filter(F.col("stars") <= 2).count(),
    "users": clean.select("user_id").distinct().count(),
    "first_year": clean.agg(F.min("year")).first()[0],
    "last_year": clean.agg(F.max("year")).first()[0],
    "input_partitions": rev_raw.rdd.getNumPartitions(),
}
save_csv(restaurants, "restaurants.csv")
with open(results_path("dataset_summary.json"), "w") as f:
    json.dump(stats, f, indent=2)

log_run("01 clean", started, spark, **stats)
spark.stop()
