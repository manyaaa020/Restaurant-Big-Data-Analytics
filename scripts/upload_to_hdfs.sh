#!/bin/bash
# Upload the Yelp JSON files into HDFS and save evidence of how they are stored.
# Run inside the namenode container:   docker exec namenode bash /scripts/upload_to_hdfs.sh
set -e
echo "Waiting for HDFS to leave safe mode and for 2 DataNodes..."
hdfs dfsadmin -safemode wait
for i in $(seq 1 60); do
  live=$(hdfs dfsadmin -report 2>/dev/null | grep -c "^Name:" || true)
  [ "$live" -ge 2 ] && break
  sleep 3
done
echo "Live DataNodes: $live"

hdfs dfs -mkdir -p /tastetrends/raw/business /tastetrends/raw/review
hdfs dfs -put -f /data/yelp_academic_dataset_business.json /tastetrends/raw/business/
hdfs dfs -put -f /data/yelp_academic_dataset_review.json  /tastetrends/raw/review/

mkdir -p /results/evidence
hdfs dfs -ls -R /tastetrends/raw | tee /results/evidence/hdfs_ls.txt
hdfs fsck /tastetrends/raw/review/yelp_academic_dataset_review.json -files -blocks -locations \
  | tee /results/evidence/hdfs_fsck.txt
hdfs dfsadmin -report > /results/evidence/hdfs_report.txt
echo "Evidence saved in results/evidence/"
