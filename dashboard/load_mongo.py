"""Load the Spark results into MongoDB (the serving layer for the dashboard).

Unstructured review text becomes JSON documents with structured fields added by Spark
(cuisine, city, year, sentiment, complaint theme, TF-IDF keywords), then indexed:
  text index      on text + keywords          -> keyword search ($text)
  compound index  on theme + cuisine          -> filter complaints by theme and cuisine
  single indexes  on business_id, user_id ... -> lookups

docker exec dashboard python load_mongo.py
"""
import glob
import json
import os

import pandas as pd
from pymongo import ASCENDING, DESCENDING, TEXT, MongoClient

R = os.environ.get("TT_RESULTS", "/results")
db = MongoClient(os.environ.get("MONGO_URI", "mongodb://mongodb:27017")).tastetrends


def sentiment(stars):
    return "negative" if stars <= 2 else ("neutral" if stars == 3 else "positive")


# 1. Reviews: one document per review (unstructured text + structured fields)
db.reviews.drop()
batch, n, n_text = [], 0, 0
with open(os.path.join(R, "mongo", "reviews.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        d["_id"] = d.pop("review_id")
        d["sentiment"] = sentiment(d["stars"])
        n_text += "text" in d
        batch.append(d)
        if len(batch) == 10000:
            db.reviews.insert_many(batch, ordered=False)
            n += len(batch)
            batch = []
if batch:
    db.reviews.insert_many(batch, ordered=False)
    n += len(batch)
db.reviews.create_index([("text", TEXT), ("keywords", TEXT)], weights={"keywords": 3, "text": 1},
                        name="review_text_search", default_language="english")
db.reviews.create_index([("theme", ASCENDING), ("cuisine", ASCENDING)], name="theme_cuisine")
db.reviews.create_index([("business_id", ASCENDING)], name="business")
db.reviews.create_index([("user_id", ASCENDING), ("year", ASCENDING)], name="user_year")
db.reviews.create_index([("stars", ASCENDING), ("year", ASCENDING)], name="stars_year")
print(f"reviews: {n} documents ({n_text} with text), indexes: {sorted(db.reviews.index_information())}")

# 2. Small result tables: one collection per CSV
for path in sorted(glob.glob(os.path.join(R, "*.csv"))):
    name = os.path.basename(path)[:-4]
    df = pd.read_csv(path)
    docs = json.loads(df.to_json(orient="records"))      # NaN -> null
    if name == "restaurants":
        for d in docs:
            d["_id"] = d["business_id"]
    db[name].drop()
    if docs:
        db[name].insert_many(docs)
    print(f"{name}: {len(docs)} documents")
db.recommendations.create_index([("user_id", ASCENDING), ("rank", ASCENDING)])
db.user_history.create_index([("user_id", ASCENDING)])

# 3. Run information: job log, dataset summary, HDFS evidence
db.run_log.drop()
with open(os.path.join(R, "run_log.jsonl")) as f:
    db.run_log.insert_many([json.loads(l) for l in f if l.strip()])
db.info.drop()
with open(os.path.join(R, "dataset_summary.json")) as f:
    db.info.insert_one({"_id": "dataset", **json.load(f), "text_docs": n_text})
model_path = os.path.join(R, "theme_model.json")
if os.path.exists(model_path):
    with open(model_path) as f:
        db.info.insert_one({"_id": "theme_model", **json.load(f)})
for path in glob.glob(os.path.join(R, "evidence", "*.txt")) + glob.glob(os.path.join(R, "q1_physical_plan.txt")):
    with open(path) as f:
        db.info.insert_one({"_id": os.path.basename(path)[:-4], "text": f.read()})
print("info:", [d["_id"] for d in db.info.find({}, {"_id": 1})])
print("collections:", sorted(db.list_collection_names()))
