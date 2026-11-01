#!/bin/bash
# Fallback: run the four Spark jobs on this machine (Spark local mode) instead of the Docker cluster.
# Use only if the Docker cluster will not start. Needs Java 17 and Python 3.10/3.11 (WSL Ubuntu works):
#   sudo apt install -y openjdk-17-jre-headless python3-venv
#   python3 -m venv .venv && .venv/bin/pip install pyspark==3.5.1 numpy pandas pymongo
#   bash run_local.sh
# MongoDB and the dashboard can still run in Docker:  docker compose up -d mongodb dashboard
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
mkdir -p local_data/raw/business local_data/raw/review
cp -n data/yelp_academic_dataset_business.json local_data/raw/business/ 2>/dev/null || true
cp -n data/yelp_academic_dataset_review.json local_data/raw/review/ 2>/dev/null || true
export TT_DATA="file://$PWD/local_data" TT_RESULTS="$PWD/results" PYSPARK_PYTHON="$PY"
rm -f results/run_log.jsonl
for job in 01_clean 02_sql_analytics 03_themes 04_recommender; do
  echo "== Spark job $job (local mode, 4 cores)"
  "$(dirname "$PY")/spark-submit" --master "local[4]" --driver-memory 3g --py-files jobs/common.py jobs/$job.py \
    2>&1 | tee results/log_$job.txt | grep -v " INFO " || true
done
echo "Now load MongoDB:  docker exec dashboard python load_mongo.py   then open http://localhost:8501"
