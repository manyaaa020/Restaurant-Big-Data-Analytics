"""Querying the unstructured review data in MongoDB - six example queries for the demo.

docker exec dashboard python mongo_queries.py        (also saved to results/mongo_queries.txt)
"""
import json
import os
import sys

from pymongo import MongoClient

R = os.environ.get("TT_RESULTS", "/results")
db = MongoClient(os.environ.get("MONGO_URI", "mongodb://mongodb:27017")).tastetrends
out = []


def show(title, query, rows):
    out.append(f"\n=== {title}\n{query}\n")
    for r in rows:
        out.append("  " + json.dumps(r, default=str)[:220])


def snippet(t, n=110):
    return (t or "").replace("\n", " ")[:n] + "..."


# 1. Full-text search ranked by relevance (uses the text index, with stemming)
q = {"$text": {"$search": "cold food waited"}}
rows = db.reviews.find(q, {"score": {"$meta": "textScore"}, "name": 1, "stars": 1, "text": 1}) \
                 .sort([("score", {"$meta": "textScore"})]).limit(5)
show("1. Text search: 'cold food waited' (top 5 by relevance)",
     'db.reviews.find({$text: {$search: "cold food waited"}}).sort({score: {$meta: "textScore"}}).limit(5)',
     [{"name": r["name"], "stars": r["stars"], "score": round(r["score"], 2), "text": snippet(r["text"])} for r in rows])

# 2. Text search combined with structured filters
q = {"$text": {"$search": "rude manager"}, "cuisine": "Mexican", "sentiment": "negative"}
n = db.reviews.count_documents(q)
rows = db.reviews.find(q, {"name": 1, "city": 1, "year": 1, "text": 1}).limit(3)
show(f"2. Text search + filters: 'rude manager' in negative Mexican reviews ({n} matches)",
     'db.reviews.find({$text: {$search: "rude manager"}, cuisine: "Mexican", sentiment: "negative"})',
     [{"name": r["name"], "city": r["city"], "year": r["year"], "text": snippet(r["text"])} for r in rows])

# 3. Aggregation: complaint themes for one cuisine
pipe = [{"$match": {"cuisine": "Fast Food", "theme": {"$exists": True}}},
        {"$group": {"_id": "$theme", "reviews": {"$sum": 1}}},
        {"$sort": {"reviews": -1}}]
show("3. Aggregation: complaint themes in Fast Food reviews",
     'db.reviews.aggregate([{$match: {cuisine: "Fast Food", theme: {$exists: true}}}, {$group: {_id: "$theme", reviews: {$sum: 1}}}, {$sort: {reviews: -1}}])',
     list(db.reviews.aggregate(pipe)))

# Theme names come from the clustering, so pick them from the data
themes = [d["theme"] for d in db.themes.find({}, {"theme": 1}).sort("n_reviews", -1)]
slow = next((t for t in themes if t.startswith("Slow service")), themes[0])
other = themes[-1]

# 4. Aggregation: restaurants with the most slow-service complaints
pipe = [{"$match": {"theme": slow}},
        {"$group": {"_id": {"name": "$name", "city": "$city"}, "complaints": {"$sum": 1}}},
        {"$sort": {"complaints": -1}}, {"$limit": 5}]
show(f"4. Aggregation: restaurants with the most '{slow}' reviews",
     f'db.reviews.aggregate([{{$match: {{theme: "{slow}"}}}}, {{$group: {{_id: {{name: "$name", city: "$city"}}, complaints: {{$sum: 1}}}}}}, {{$sort: {{complaints: -1}}}}, {{$limit: 5}}])',
     list(db.reviews.aggregate(pipe)))

# 5. Query on the keyword tags that Spark extracted from the text
pipe = [{"$match": {"keywords": "dirty"}},
        {"$group": {"_id": "$city", "reviews": {"$sum": 1}}}, {"$sort": {"reviews": -1}}, {"$limit": 5}]
show("5. Keyword tags: negative reviews tagged 'dirty', by city",
     'db.reviews.aggregate([{$match: {keywords: "dirty"}}, {$group: {_id: "$city", reviews: {$sum: 1}}}, {$sort: {reviews: -1}}, {$limit: 5}])',
     list(db.reviews.aggregate(pipe)))

# 6. Index use: explain() shows which index MongoDB chose
def stages(plan):
    names, node = [], plan
    while node:
        names.append(node.get("stage") + (f"({node['indexName']})" if "indexName" in node else ""))
        node = node.get("inputStage") or (node.get("inputStages") or [None])[0]
    return " <- ".join(names)


p1 = db.reviews.find({"theme": other, "cuisine": "American"}).explain()["queryPlanner"]["winningPlan"]
p2 = db.reviews.find({"$text": {"$search": "rude manager"}}).explain()["queryPlanner"]["winningPlan"]
show("6. Index use (explain)", f"db.reviews.find({{theme: '{other}', cuisine: 'American'}}).explain()  and  $text query .explain()",
     [{"theme + cuisine filter": stages(p1.get("queryPlan", p1))},
      {"text search": stages(p2.get("queryPlan", p2))}])

text = "\n".join(out)
print(text)
if os.path.isdir(R) and os.access(R, os.W_OK):
    with open(os.path.join(R, "mongo_queries.txt"), "w") as f:
        f.write(text)
