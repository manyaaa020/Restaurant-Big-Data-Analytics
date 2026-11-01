"""Step 2 - Analytical queries with Spark SQL.

Registers the cleaned Parquet data as two SQL views and runs the five queries in jobs/sql/.
Each result is saved as a CSV; the physical plan of Q1 is saved to show how Spark runs a join.
"""
import glob
import os
import time

from common import DATA, SQL_DIR, spark_session, save_csv, results_path, log_run, year_window, update_summary

started = time.time()
spark = spark_session("tastetrends-02-sql")

spark.read.parquet(f"{DATA}/clean/restaurants").createOrReplaceTempView("restaurants")
(spark.read.parquet(f"{DATA}/clean/reviews")
      .withColumnRenamed("year", "review_year")
      .createOrReplaceTempView("reviews"))

first_year, last_full, _ = year_window(spark.table("reviews").withColumnRenamed("review_year", "year"))
update_summary(trend_first_year=first_year, trend_last_year=last_full)

rows = {}
for path in sorted(glob.glob(os.path.join(SQL_DIR, "q*.sql"))):
    name = os.path.basename(path)[:-4]                 # e.g. q1_city_overview
    with open(path) as f:
        sql = f.read().replace("{FIRST_YEAR}", str(first_year)).replace("{LAST_FULL_YEAR}", str(last_full))
    df = spark.sql(sql)
    if name.startswith("q1"):
        plan = df._jdf.queryExecution().executedPlan().toString()
        with open(results_path("q1_physical_plan.txt"), "w") as f:
            f.write(plan)
    t = time.time()
    rows[name] = len(save_csv(df, f"{name}.csv"))
    print(f"{name}: {rows[name]} rows in {time.time() - t:.1f} s")

log_run("02 spark sql", started, spark, **{f"rows_{k[:2]}": v for k, v in rows.items()})
spark.stop()
