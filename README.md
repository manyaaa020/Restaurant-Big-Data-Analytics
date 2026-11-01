# TasteTrends: Restaurant complaint mining with Apache Spark

Finds **why** restaurant customers complain, using 229,907 Yelp reviews, a Spark cluster and MongoDB.

```
Yelp JSON ──► HDFS (2 DataNodes, 128 MB blocks, 2 copies)
                │
                ▼
        Spark standalone cluster (master + 2 workers)
          01 clean        PySpark: keep restaurant reviews, add cuisine, clean text  → Parquet in HDFS
          02 spark sql    Spark SQL: 5 analytical queries (joins, RANK, LAG, HAVING)
          03 themes       MLlib: TF-IDF + K-Means → 8 complaint themes
          04 recommender  MLlib: ALS → top-10 restaurants per user, evaluated on future reviews
                │
                ▼
        MongoDB  (158k review documents + results, text index)  ──►  Streamlit dashboard
```
Everything runs in Docker Compose: 8 containers.

## What you need

- Docker Desktop (Windows: WSL 2 backend) with at least **8 GB memory** (Settings → Resources; the default on a 16 GB laptop is fine) and ~5 GB free disk.
- The two Yelp files, copied into the `data` folder with exactly these names:
  - `data/yelp_academic_dataset_business.json`
  - `data/yelp_academic_dataset_review.json`

## Run it (one command)

Open PowerShell in this folder and run:
```
powershell -ExecutionPolicy Bypass -File run_all.ps1
```
(In WSL / Git Bash: `bash run_all.sh`)

The first run downloads the images and builds two of them (10-15 minutes, needs internet). Then it:
1. starts the 8 containers,
2. uploads the dataset to HDFS and saves `hdfs fsck` evidence in `results/evidence/`,
3. runs the four Spark jobs on the cluster (about 5-8 minutes in total),
4. loads MongoDB and runs the example queries (`results/mongo_queries.txt`).

Then open:

| What | Address |
|---|---|
| Dashboard | http://localhost:8501 |
| HDFS NameNode (Datanodes tab, Utilities → Browse) | http://localhost:9870 |
| Spark master (workers, completed applications) | http://localhost:8080 |
| A Spark job while it runs (jobs, stages, DAG, executors) | http://localhost:4040 |

## Run one step by hand (for the demo)

```
docker exec namenode hdfs fsck /tastetrends/raw/review/yelp_academic_dataset_review.json -files -blocks -locations
docker exec spark-master spark-submit --master spark://spark-master:7077 --driver-memory 1536m --executor-memory 1g --total-executor-cores 4 --py-files /opt/jobs/common.py /opt/jobs/02_sql_analytics.py
docker exec dashboard python mongo_queries.py
docker exec -it mongodb mongosh tastetrends --eval "db.reviews.find({\$text: {\$search: 'rude manager'}}).limit(2)"
```

## Files

| Path | What it does |
|---|---|
| `docker-compose.yml`, `hadoop.env` | The 8 containers and HDFS settings (replication 2, 128 MB blocks) |
| `docker/spark/Dockerfile` | Spark 3.5.1 image + numpy/pandas |
| `scripts/upload_to_hdfs.sh` | Puts the JSON files into HDFS, saves fsck/report evidence |
| `jobs/common.py` | Shared settings: paths, schemas, cuisine rules, run log |
| `jobs/01_clean.py` | ETL: filter restaurants, broadcast join, clean text, write Parquet to HDFS |
| `jobs/02_sql_analytics.py` + `jobs/sql/` | Spark SQL queries Q1-Q5 |
| `jobs/03_themes.py` | TF-IDF + K-Means complaint themes, exports review documents for MongoDB |
| `jobs/04_recommender.py` | ALS recommender + evaluation (hit rate, precision, recall, coverage) |
| `dashboard/load_mongo.py` | Loads reviews and results into MongoDB, creates the indexes |
| `dashboard/mongo_queries.py` | Six example MongoDB queries (text search, aggregation, explain) |
| `dashboard/app.py` | Streamlit dashboard (6 pages): overview, complaint themes (filters + click to read reviews), Spark SQL insights, ALS recommendations, MongoDB search + query lab, cluster and pipeline |
| `results/` | Output of your run (CSV, JSON, logs, evidence) |
| `results_reference/` | Output of our test run, for comparison |

## Running on the full Yelp dataset (real big data)

The code works unchanged on the current Yelp Open Dataset (`yelp_dataset.tar`, about 7 million
reviews, 5.3 GB review file). Year windows and the recommender's train/test cutoff are taken from
the data. When there are more than 500,000 restaurant reviews, MongoDB keeps the full text only for
the complaint (1-2 star) reviews, so it stays laptop-sized.

1. Free disk space: about 30 GB (HDFS stores every block twice).
2. Docker memory: 10 GB if possible. Create `C:\Users\<you>\.wslconfig` with
   ```
   [wsl2]
   memory=10GB
   ```
   then run `wsl --shutdown` and restart Docker Desktop.
3. Move the 2013 files out of `data` (for example into `data_2013`), then extract the two files
   from the tar into `data` (PowerShell, in the project folder):
   ```
   tar -xf C:\path\to\yelp_dataset.tar -C data yelp_academic_dataset_business.json yelp_academic_dataset_review.json
   ```
4. Run `run_all.ps1` as before. Expect roughly an hour: most of it is the HDFS upload, the themes job
   and loading MongoDB.

## If something goes wrong

- **A container keeps restarting:** `docker compose logs <name>`. Usually memory: raise Docker's memory limit, or stop other apps.
- **"Missing data/..."**: the Yelp files are not in `data/` or have different names.
- **HDFS upload hangs at "Waiting"**: a DataNode did not start. `docker compose restart datanode1 datanode2`, then run `docker exec namenode bash /scripts/upload_to_hdfs.sh` again.
- **A Spark job fails:** read `results/log_<job>.txt`. Re-run just that job with the `spark-submit` line above.
- **Docker will not work at all:** `run_local.sh` runs the same four jobs in Spark local mode (see the comments at its top), and `docker compose up -d mongodb dashboard` still gives you MongoDB and the dashboard.
- **Start again from zero:** `docker compose down -v` (deletes HDFS and MongoDB data), then run `run_all.ps1` again.

Results are reproducible: the theme clustering uses fixed seeds and hash partitioning, so a rerun on any machine gives the same themes and numbers as `results_reference/`.
