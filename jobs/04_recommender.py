"""Step 4 - Restaurant recommender with Spark MLlib ALS, evaluated on future reviews.

Time-based split: reviews before the last full year train the model; reviews from that year on
are hidden (2012 for the 2013 edition; see year_window in common.py).
Test users: >= 5 training reviews and >= 1 hidden review.
Three methods are scored the same way (top 10 new restaurants per test user):
  ALS implicit - each review counts as a visit (confidence 1 + alpha), the main model
  ALS explicit - predicts the star rating
  Popularity   - the 10 most-reviewed restaurants the user has not visited
Metrics: hit rate@10 (users with >= 1 correct), precision@10, recall@10, catalogue coverage.
"""
import time

import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import Window, functions as F
from common import (DATA, MIN_TRAIN, N_EVAL_USERS, spark_session, save_csv,
                    results_path, log_run, year_window, update_summary)

started = time.time()
spark = spark_session("tastetrends-04-recommender")

r = spark.read.parquet(f"{DATA}/clean/reviews").select("user_id", "business_id", "stars", "year")
spark.conf.set("spark.sql.shuffle.partitions", max(8, r.count() // 100000))   # scales with the data; 8 for 2013
_, _, CUTOFF_YEAR = year_window(r)
update_summary(cutoff_year=CUTOFF_YEAR)
print(f"train: reviews before {CUTOFF_YEAR}; test: {CUTOFF_YEAR} onwards")
train = (r.filter(F.col("year") < CUTOFF_YEAR)
          .groupBy("user_id", "business_id").agg(F.avg("stars").alias("stars")).cache())
test = r.filter(F.col("year") >= CUTOFF_YEAR).select("user_id", "business_id").distinct()

active = train.groupBy("user_id").count().filter(F.col("count") >= MIN_TRAIN).select("user_id")
eval_users = (active.join(test.select("user_id").distinct(), "user_id")
              .orderBy("user_id").limit(N_EVAL_USERS).cache())

# ALS needs integer ids
u_idx = StringIndexer(inputCol="user_id", outputCol="uid").fit(train)
i_idx = StringIndexer(inputCol="business_id", outputCol="iid").fit(train)
train_ix = i_idx.transform(u_idx.transform(train)).cache()
eval_ix = u_idx.transform(eval_users).select(F.col("uid").cast("int").alias("uid"), "user_id")
items = spark.createDataFrame(list(enumerate(i_idx.labels)), ["iid", "business_id"])
seen = train.select("user_id", "business_id")


def top10(implicit):
    als = ALS(userCol="uid", itemCol="iid", ratingCol="stars", rank=20, maxIter=10, regParam=0.1,
              implicitPrefs=implicit, alpha=20.0, nonnegative=True, coldStartStrategy="drop", seed=42)
    model = als.fit(train_ix)
    w = Window.partitionBy("user_id").orderBy("pos")
    return (model.recommendForUserSubset(eval_ix.select("uid"), 50)
            .select("uid", F.posexplode("recommendations").alias("pos", "rec"))
            .select(F.col("uid").cast("int").alias("uid"), "pos",
                    F.col("rec.iid").alias("iid"), F.round("rec.rating", 4).alias("score"))
            .join(eval_ix, "uid").join(items, "iid")
            .join(seen, ["user_id", "business_id"], "left_anti")        # only new restaurants
            .withColumn("rank", F.row_number().over(w)).filter("rank <= 10")
            .select("user_id", "business_id", "rank", "score"))


t0 = time.time()
implicit_recs = top10(True).cache()
n_rec = implicit_recs.count()
t_implicit = round(time.time() - t0, 1)
explicit_recs = top10(False)

pop = train.groupBy("business_id").count().orderBy(F.desc("count"), "business_id").limit(200)
w = Window.partitionBy("user_id").orderBy(F.desc("count"), "business_id")
pop_recs = (eval_users.crossJoin(pop).join(seen, ["user_id", "business_id"], "left_anti")
            .withColumn("rank", F.row_number().over(w)).filter("rank <= 10")
            .select("user_id", "business_id", "rank"))

# Scoring on the driver (2,000-5,000 users x 10 rows)
test_pd = test.join(eval_users, "user_id").toPandas()
truth = test_pd.groupby("user_id")["business_id"].apply(set)
n_items = train.select("business_id").distinct().count()


def score(recs_pd, name):
    lists = recs_pd.sort_values("rank").groupby("user_id")["business_id"].apply(list)
    hits, prec, rec = [], [], []
    for u, actual in truth.items():
        top = lists.get(u, [])[:10]
        h = len(set(top) & actual)
        hits.append(h > 0)
        prec.append(h / 10)
        rec.append(h / len(actual))
    n = len(truth)
    return {"method": name, "test_users": n, "hit_rate_at_10": round(sum(hits) / n, 4),
            "precision_at_10": round(sum(prec) / n, 4), "recall_at_10": round(sum(rec) / n, 4),
            "coverage": round(recs_pd["business_id"].nunique() / n_items, 4)}


imp_pd = implicit_recs.toPandas()
metrics = pd.DataFrame([score(imp_pd, "ALS implicit (reviews as visits)"),
                        score(explicit_recs.toPandas(), "ALS explicit (star ratings)"),
                        score(pop_recs.toPandas(), "Popularity baseline")])
metrics.to_csv(results_path("rec_metrics.csv"), index=False)
print(metrics.to_string(index=False))

names = spark.read.parquet(f"{DATA}/clean/restaurants").select("business_id", "name", "city", "cuisine")
save_csv(implicit_recs.join(names, "business_id").orderBy("user_id", "rank"), "recommendations.csv")
save_csv(seen.join(eval_users, "user_id").join(names, "business_id").orderBy("user_id", "name"),
         "user_history.csv")

m = metrics.iloc[0]
log_run("04 recommender", started, spark, cutoff_year=CUTOFF_YEAR, train_pairs=train.count(), test_users=int(m["test_users"]),
        items=n_items, als_train_seconds=t_implicit, hit_rate_at_10=float(m["hit_rate_at_10"]))
spark.stop()
