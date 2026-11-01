"""TasteTrends dashboard (Streamlit). Everything on it is read from MongoDB, much of it live."""
import json
import math
import os
import re
import time

import altair as alt
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ExecutionTimeout

st.set_page_config(page_title="TasteTrends", layout="wide")
BLUE, RED, GREY = "#2f6f9f", "#c0583c", "#b8b8b0"


@st.cache_resource
def database():
    return MongoClient(os.environ.get("MONGO_URI", "mongodb://mongodb:27017")).tastetrends


db = database()


@st.cache_data(ttl=600)
def table(name, sort=None):
    df = pd.DataFrame(list(db[name].find({}, {"_id": 0})))
    return df.sort_values(sort) if sort and not df.empty else df


@st.cache_data(ttl=600)
def agg(pipeline_json):
    """Run an aggregation on the review documents (cached by the pipeline text)."""
    return list(db.reviews.aggregate(json.loads(pipeline_json), allowDiskUse=True))


@st.cache_data(ttl=3600)
def distinct(field):
    """Values of a field, read from the small summary table rather than scanning every review."""
    vals = db.review_cube.distinct(field) or db.reviews.distinct(field)
    return sorted(v for v in vals if v not in (None, ""))


def need(df, what):
    if df.empty:
        st.info(f"No {what} in MongoDB yet. Run the Spark jobs and load_mongo.py first.")
        return True
    return False


def bar(df, x, y, color=BLUE, fmt=None, sort="-x", title=None, select=None):
    enc_x = alt.X(f"{x}:Q", title=title, axis=alt.Axis(format=fmt) if fmt else alt.Axis())
    ch = alt.Chart(df).mark_bar(color=color).encode(
        x=enc_x, y=alt.Y(f"{y}:N", sort=sort, title=None, axis=alt.Axis(labelLimit=260)),
        tooltip=list(df.columns))
    if select is not None:
        ch = ch.add_params(select).encode(opacity=alt.condition(select, alt.value(1), alt.value(0.45)))
    return ch


def show_review(r, chars=500):
    head = f"**{r.get('name', '')}** ({r.get('city', '')}, {r.get('year', '')}) · {r.get('stars', '')} star"
    if r.get("theme"):
        head += f" · {r['theme']}"
    if r.get("keywords"):
        head += f" · keywords: _{', '.join(r['keywords'])}_"
    st.markdown(head)
    t = r.get("text", "")
    st.write(t[:chars] + ("..." if len(t) > chars else ""))


info = db.info.find_one({"_id": "dataset"}) or {}
CUTOFF = info.get("cutoff_year", 2012)
st.sidebar.title("TasteTrends")
st.sidebar.caption("What restaurant customers complain about, mined from Yelp reviews with Apache Spark")
page = st.sidebar.radio("Page", ["Overview", "Complaint themes", "Spark SQL insights",
                                 "Recommendations", "MongoDB search", "Cluster and pipeline"])
st.sidebar.divider()
st.sidebar.caption("Yelp JSON → HDFS → Spark (SQL, MLlib) → MongoDB → Streamlit")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("TasteTrends")
    st.markdown("**Problem.** A restaurant's star rating says *that* customers are unhappy, not *why*. "
                "The reasons are buried in the text of millions of reviews, which is too much to read by hand. "
                "TasteTrends uses Spark to group 1-2 star Yelp reviews into complaint themes, compare them across "
                "cuisines, cities and years, and recommend restaurants, with MongoDB serving the results.")
    st.code("Yelp JSON → HDFS (2 DataNodes, 2 copies) → Spark cluster: clean → Spark SQL → MLlib (K-Means, ALS)\n"
            "          → MongoDB (review documents + text index) → this dashboard", language="text")
    if not info:
        st.info("No data loaded yet. Run the Spark jobs and load_mongo.py first.")
        st.stop()
    c = st.columns(4)
    c[0].metric("Reviews in the dataset", f"{info['raw_reviews']:,}")
    c[1].metric("Restaurants", f"{info['restaurants']:,}")
    c[2].metric("Restaurant reviews analysed", f"{info['restaurant_reviews']:,}")
    c[3].metric("Negative (1-2 star) reviews", f"{info['negative_reviews']:,}")
    st.caption(f"Yelp reviews from {info['first_year']} to {info['last_year']}, {info['users']:,} reviewers.")
    q1 = table("q1_city_overview")
    if not need(q1, "city results"):
        st.subheader("Reviews and complaint rate by city")
        left, right = st.columns([3, 2])
        left.altair_chart(bar(q1, "n_reviews", "city", title="Restaurant reviews"), use_container_width=True)
        right.dataframe(q1.rename(columns={"n_reviews": "reviews", "avg_stars": "avg stars",
                                           "neg_share": "share 1-2 star"}), hide_index=True)

# ---------------------------------------------------------------- Complaint themes
elif page == "Complaint themes":
    st.title("What customers complain about")
    st.caption("1-2 star reviews grouped by Spark MLlib (TF-IDF + K-Means, cosine distance). Filter, then click a "
               "theme to read its reviews. Counts come from review_cube, a summary table Spark pre-aggregated.")
    cube = table("review_cube")
    if need(cube, "summary table (review_cube)"):
        st.stop()
    years = sorted(cube["year"].dropna().astype(int).unique())
    f1, f2, f3 = st.columns([2, 2, 3])
    cities = f1.multiselect("City", cube.groupby("city")["count"].sum().sort_values(ascending=False).index.tolist(),
                            placeholder="All cities")
    cuisines = f2.multiselect("Cuisine", sorted(cube["cuisine"].dropna().unique()), placeholder="All cuisines")
    yr = f3.slider("Years", int(years[0]), int(years[-1]), (int(years[0]), int(years[-1])))
    sel = cube[(cube["year"] >= yr[0]) & (cube["year"] <= yr[1])]
    if cities:
        sel = sel[sel["city"].isin(cities)]
    if cuisines:
        sel = sel[sel["cuisine"].isin(cuisines)]
    total = int(sel["count"].sum())
    if total == 0:
        st.warning("No reviews match these filters.")
        st.stop()
    neg = int(sel[sel["stars"] <= 2]["count"].sum())
    c = st.columns(3)
    c[0].metric("Reviews", f"{total:,}")
    c[1].metric("Average stars", f"{(sel['stars'] * sel['count']).sum() / total:.2f}")
    c[2].metric("Share 1-2 star", f"{neg / total:.1%}")

    th = sel.dropna(subset=["theme"]).groupby("theme", as_index=False)["count"].sum().rename(columns={"count": "reviews"})
    if th.empty:
        st.info("No clustered complaints in this selection.")
        st.stop()
    th["share"] = th["reviews"] / th["reviews"].sum()
    terms = table("themes")
    if not terms.empty:
        th = th.merge(terms[["theme", "top_terms"]], on="theme", how="left")
    pick = alt.selection_point(fields=["theme"], name="pick")
    ev = st.altair_chart(bar(th, "reviews", "theme", color=RED, title="Negative reviews", select=pick),
                         use_container_width=True, on_select="rerun", key="theme_bar")
    chosen = [p["theme"] for p in (ev.selection.get("pick") or []) if "theme" in p] if ev else []
    if chosen:
        match = {"theme": chosen[0], "year": {"$gte": yr[0], "$lte": yr[1]}}
        if cities:
            match["city"] = {"$in": cities}
        if cuisines:
            match["cuisine"] = {"$in": cuisines}
        a, b = st.columns([2, 3])
        t0 = time.time()
        kw = pd.DataFrame(agg(json.dumps([{"$match": match}, {"$unwind": "$keywords"},
                                          {"$group": {"_id": "$keywords", "reviews": {"$sum": 1}}},
                                          {"$sort": {"reviews": -1}}, {"$limit": 15}])))
        with a:
            st.markdown(f"**Top keywords: {chosen[0]}**")
            if not kw.empty:
                st.dataframe(kw.rename(columns={"_id": "keyword"}), hide_index=True, height=420)
        with b:
            st.markdown(f"**Reviews about: {chosen[0]}**")
            for r in db.reviews.find(match).limit(4):
                show_review(r, 350)
        st.code("db.reviews.aggregate([{$match: " + json.dumps(match) + "}, {$unwind: '$keywords'}, "
                f"{{$group: ...}}])   # {time.time() - t0:.2f} s", language="javascript")
    else:
        st.markdown("**Click a theme bar to see its top keywords and reviews.**")

    by = table("theme_by_year")
    if not by.empty:
        st.subheader("Themes over time (all cities and cuisines)")
        prefixes = ["Slow service", "Rude staff", "Bar, noise"]
        default = [next(t for t in sorted(by["theme"].unique()) if t.startswith(p))
                   for p in prefixes if any(t.startswith(p) for t in by["theme"].unique())]
        pick_t = st.multiselect("Themes", sorted(by["theme"].unique()), default=default or sorted(by["theme"].unique())[:3])
        st.altair_chart(alt.Chart(by[by["theme"].isin(pick_t)]).mark_line(point=True).encode(
            x=alt.X("year:O", title="Year"), y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="Share of that year's complaints"),
            color=alt.Color("theme:N", legend=alt.Legend(orient="bottom", columns=1, labelLimit=320)),
            tooltip=["theme", "year", "count", "share"]), use_container_width=True)

# ---------------------------------------------------------------- Spark SQL
elif page == "Spark SQL insights":
    st.title("Ratings by cuisine, city and year (Spark SQL)")
    q2 = table("q2_cuisine_summary")
    if not need(q2, "cuisine results"):
        st.subheader("Share of 1-2 star reviews by cuisine")
        st.altair_chart(bar(q2, "neg_share", "cuisine", color=RED, fmt="%", title="Share of reviews with 1-2 stars"),
                        use_container_width=True)
    q3 = table("q3_cuisine_rank_by_city")
    if not q3.empty:
        st.subheader("Best-rated cuisines in each city (RANK)")
        city = st.selectbox("City", sorted(q3["city"].unique()))
        st.dataframe(q3[q3["city"] == city].sort_values("rnk"), hide_index=True)
    q4 = table("q4_cuisine_trend")
    if not q4.empty:
        st.subheader("Average rating per year (LAG gives the yearly change)")
        opts = sorted(q4["cuisine"].unique())
        pick = st.multiselect("Cuisines", opts, default=[c for c in ["Mexican", "Italian", "Fast Food"] if c in opts])
        st.altair_chart(alt.Chart(q4[q4["cuisine"].isin(pick)]).mark_line(point=True).encode(
            x=alt.X("review_year:O", title="Year"), y=alt.Y("avg_stars:Q", scale=alt.Scale(zero=False), title="Average stars"),
            color="cuisine:N", tooltip=["cuisine", "review_year", "n_reviews", "avg_stars", "yoy_change"]),
            use_container_width=True)
    q5 = table("q5_most_complained")
    if not q5.empty:
        st.subheader("Restaurants with the highest share of 1-2 star reviews (at least 200 reviews)")
        st.dataframe(q5.drop(columns=["business_id"]).head(15), hide_index=True)

# ---------------------------------------------------------------- Recommendations
elif page == "Recommendations":
    st.title("Restaurant recommendations (Spark ALS)")
    m = table("rec_metrics")
    if need(m, "recommender results"):
        st.stop()
    m = table("rec_metrics")
    if need(m, "recommender results"):
        st.stop()
    st.caption(f"Trained on reviews before {CUTOFF}; tested on whether each user later reviewed one of their top 10.")
    st.dataframe(m.rename(columns={"hit_rate_at_10": "hit rate @10", "precision_at_10": "precision @10",
                                   "recall_at_10": "recall @10"}), hide_index=True)
    users = sorted(db.recommendations.distinct("user_id"))
    hits_only = st.checkbox("Only show users where a recommendation came true")
    if hits_only:
        later = {(d["user_id"], d["business_id"]) for d in db.reviews.find({"year": {"$gte": CUTOFF}, "user_id": {"$in": users}},
                                                                            {"user_id": 1, "business_id": 1})}
        recs_all = list(db.recommendations.find({}, {"user_id": 1, "business_id": 1}))
        users = sorted({d["user_id"] for d in recs_all if (d["user_id"], d["business_id"]) in later})
    user = st.selectbox(f"Test user ({len(users):,})", users)
    later_ids = {d["business_id"] for d in db.reviews.find({"user_id": user, "year": {"$gte": CUTOFF}}, {"business_id": 1})}
    left, right = st.columns(2)
    hist = pd.DataFrame(list(db.user_history.find({"user_id": user}, {"_id": 0, "name": 1, "city": 1, "cuisine": 1})))
    left.subheader(f"Reviewed before {CUTOFF}")
    left.dataframe(hist, hide_index=True)
    recs = pd.DataFrame(list(db.recommendations.find({"user_id": user}, {"_id": 0}).sort("rank", 1)))
    if not recs.empty:
        recs["visited later"] = recs["business_id"].isin(later_ids).map({True: "yes", False: ""})
        recs = recs[["rank", "name", "city", "cuisine", "score", "visited later"]]
    right.subheader("Top 10 recommended")
    right.dataframe(recs, hide_index=True)

# ---------------------------------------------------------------- MongoDB
elif page == "MongoDB search":
    st.title("MongoDB: search and query the reviews")
    search_tab, lab_tab = st.tabs(["Full-text search", "Query lab"])
    with search_tab:
        n_text = db.info.find_one({"_id": "dataset"}, {"text_docs": 1}) or {}
        st.caption(f"MongoDB full-text search over {n_text.get('text_docs', db.reviews.estimated_document_count()):,} "
                   "review documents with text (text index with stemming, ranked by relevance).")
        st.caption("Tip: specific words (cockroach, poisoning, refund) return quickly; very common words "
                   "(food, service) match hundreds of thousands of reviews and take longer to rank.")
        with st.form("search"):
            words = st.text_input("Words to search for", "cockroach")
            c = st.columns(4)
            sentiment = c[0].selectbox("Sentiment", ["any", "negative", "neutral", "positive"], index=1)
            cuisine = c[1].selectbox("Cuisine", ["any"] + distinct("cuisine"))
            theme = c[2].selectbox("Complaint theme", ["any"] + distinct("theme"))
            phrase = c[3].checkbox("Exact phrase", help="Wraps the words in quotes, so MongoDB matches them together")
            go = st.form_submit_button("Search", type="primary")
        if go and words.strip():
            q = {"$text": {"$search": f'"{words}"' if phrase else words}}
            for key, val in [("sentiment", sentiment), ("cuisine", cuisine), ("theme", theme)]:
                if val != "any":
                    q[key] = val
            st.code("db.reviews.find(" + json.dumps(q) + ")\n  .sort({score: {$meta: 'textScore'}}).limit(10)", language="javascript")
            t0 = time.time()
            try:
                n = db.reviews.count_documents(q, limit=1001, maxTimeMS=20000)
                docs = list(db.reviews.find(q, {"score": {"$meta": "textScore"}, "name": 1, "city": 1, "stars": 1, "year": 1,
                                                "theme": 1, "keywords": 1, "text": 1})
                            .sort([("score", {"$meta": "textScore"})]).limit(10).max_time_ms(30000))
            except ExecutionTimeout:
                st.warning("This search matches too many reviews to rank quickly. Add a filter or use more specific words.")
                st.stop()
            shown = "more than 1,000" if n > 1000 else f"{n:,}"
            st.write(f"**{shown}** matching reviews ({time.time() - t0:.2f} s), top 10 by relevance:")
            for r in docs:
                show_review(r, 400)
    with lab_tab:
        st.caption("Run your own find or aggregate on the review documents (read-only, at most 50 results), "
                   "and see how MongoDB executed it.")
        theme_list = table("themes", sort="n_reviews")["theme"].tolist()[::-1] or ["Slow service and long waits"]
        slow = next((t for t in theme_list if t.startswith("Slow service")), theme_list[0])
        second = theme_list[1] if len(theme_list) > 1 else theme_list[0]
        templates = {
            "Text search": ("find", '{"$text": {"$search": "rude manager"}, "cuisine": "Mexican"}'),
            "Theme + cuisine (uses compound index)": ("find", json.dumps({"theme": slow, "cuisine": "Pizza"})),
            "Complaints per theme for a city": ("aggregate", '[{"$match": {"city": "Tempe", "theme": {"$exists": true}}},\n {"$group": {"_id": "$theme", "reviews": {"$sum": 1}}},\n {"$sort": {"reviews": -1}}]'),
            "Top keywords in 1-star reviews": ("aggregate", '[{"$match": {"stars": 1}}, {"$unwind": "$keywords"},\n {"$group": {"_id": "$keywords", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}, {"$limit": 15}]'),
            f"Restaurants with most '{second}' complaints": ("aggregate", '[{"$match": {"theme": ' + json.dumps(second) + '}},\n {"$group": {"_id": "$name", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}, {"$limit": 10}]'),
        }
        name = st.selectbox("Start from a template", list(templates))
        kind, body = templates[name]
        kind = st.radio("Operation", ["find", "aggregate"], index=0 if kind == "find" else 1, horizontal=True)
        text = st.text_area("Filter (find) or pipeline (aggregate), as JSON", body, height=140, key=f"q_{name}_{kind}")
        if st.button("Run query", type="primary"):
            try:
                q = json.loads(text)
                if any(k in text for k in ['"$out"', '"$merge"', '"$where"', '"$function"']):
                    raise ValueError("Write and JavaScript stages are not allowed here.")
                t0 = time.time()
                if kind == "find":
                    proj = {"text": 0}
                    docs = list(db.reviews.find(q, proj).limit(50).max_time_ms(30000))
                    ex = db.reviews.find(q, proj).limit(50).explain()
                    stats = ex.get("executionStats", {})
                    plan = ex["queryPlanner"]["winningPlan"]
                    plan = plan.get("queryPlan", plan)
                    stages, node = [], plan
                    while node:
                        stages.append(node.get("stage", "") + (f"({node['indexName']})" if "indexName" in node else ""))
                        node = node.get("inputStage") or (node.get("inputStages") or [None])[0]
                    c = st.columns(3)
                    c[0].metric("Results (max 50)", len(docs))
                    c[1].metric("Time", f"{(time.time() - t0) * 1000:.0f} ms")
                    c[2].metric("Plan", stages[-1] if stages else "?")
                    st.caption("Winning plan: " + " ← ".join(stages))
                else:
                    if not isinstance(q, list):
                        raise ValueError("An aggregate pipeline must be a JSON list of stages.")
                    docs = list(db.reviews.aggregate(q + [{"$limit": 50}], allowDiskUse=True, maxTimeMS=30000))
                    st.metric("Time", f"{(time.time() - t0) * 1000:.0f} ms")
                df = pd.json_normalize(docs)
                st.dataframe(df.astype(str), hide_index=True)
            except Exception as e:  # show any JSON or query error to the user
                st.error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- Pipeline
else:
    st.title("Cluster and pipeline")
    log = table("run_log", sort="job")
    if not need(log, "run log"):
        st.subheader("Spark jobs")
        cols = [c for c in ["job", "seconds", "master", "executors", "default_parallelism", "finished"] if c in log]
        st.dataframe(log[cols], hide_index=True)
    dist = table("partition_distribution")
    if not dist.empty:
        st.subheader("Where the review file was processed")
        st.caption("Each input partition of the review JSON and the Spark worker that read it.")
        a, b = st.columns([2, 3])
        a.dataframe(dist, hide_index=True)
        per = dist.groupby("worker", as_index=False)["rows"].sum()
        b.altair_chart(alt.Chart(per).mark_bar(color=BLUE).encode(
            x=alt.X("rows:Q", title="Reviews read"), y=alt.Y("worker:N", title=None), tooltip=["worker", "rows"]),
            use_container_width=True)
    for key, title in [("hdfs_fsck", "HDFS: blocks and replicas of the review file"),
                       ("hdfs_report", "HDFS: DataNodes"), ("q1_physical_plan", "Spark SQL physical plan for Q1")]:
        doc = db.info.find_one({"_id": key})
        if doc:
            with st.expander(title):
                st.code(doc["text"][:6000])
    st.subheader("MongoDB collections")
    st.dataframe(pd.DataFrame([{"collection": n, "documents": db[n].estimated_document_count(),
                                "indexes": ", ".join(sorted(db[n].index_information()))}
                               for n in sorted(db.list_collection_names())]), hide_index=True)
