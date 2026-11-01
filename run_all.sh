#!/bin/bash
# Runs the whole pipeline on the Docker cluster. From the project folder:  bash run_all.sh
set -e
cd "$(dirname "$0")"
for f in data/yelp_academic_dataset_business.json data/yelp_academic_dataset_review.json; do
  [ -f "$f" ] || { echo "Missing $f - copy the Yelp files into the data folder first."; exit 1; }
done

echo "== 1/6 Starting the containers"
docker compose up -d --build
sleep 20

echo "== 2/6 Uploading the dataset to HDFS"
docker exec namenode bash /scripts/upload_to_hdfs.sh

SUBMIT="/opt/spark/bin/spark-submit --master spark://spark-master:7077 --driver-memory 1536m \
  --executor-memory 1g --total-executor-cores 4 --py-files /opt/jobs/common.py"
rm -f results/run_log.jsonl
for job in 01_clean 02_sql_analytics 03_themes 04_recommender; do
  echo "== 3/6 Spark job $job"
  docker exec spark-master $SUBMIT /opt/jobs/$job.py 2>&1 | tee results/log_$job.txt | grep -v " INFO " || true
  grep -q "\"job\": \"${job:0:2}" results/run_log.jsonl || { echo "Job $job failed - see results/log_$job.txt"; exit 1; }
done

echo "== 4/6 Loading MongoDB"
docker exec dashboard python load_mongo.py
echo "== 5/6 MongoDB example queries"
docker exec dashboard python mongo_queries.py
echo "== 6/6 Done. Dashboard: http://localhost:8501   Spark: http://localhost:8080   HDFS: http://localhost:9870"
