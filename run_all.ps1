# Runs the whole pipeline on the Docker cluster (Windows PowerShell). From the project folder:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
foreach ($f in "data\yelp_academic_dataset_business.json", "data\yelp_academic_dataset_review.json") {
    if (-not (Test-Path $f)) { Write-Host "Missing $f - copy the Yelp files into the data folder first."; exit 1 }
}

Write-Host "== 1/6 Starting the containers"
docker compose up -d --build
Start-Sleep -Seconds 20

Write-Host "== 2/6 Uploading the dataset to HDFS"
docker exec namenode bash /scripts/upload_to_hdfs.sh

$submit = @("/opt/spark/bin/spark-submit", "--master", "spark://spark-master:7077", "--driver-memory", "1536m",
            "--executor-memory", "1g", "--total-executor-cores", "4", "--py-files", "/opt/jobs/common.py")
Remove-Item results\run_log.jsonl -ErrorAction SilentlyContinue
$ErrorActionPreference = "Continue"     # Spark writes its progress log to stderr
foreach ($job in "01_clean", "02_sql_analytics", "03_themes", "04_recommender") {
    Write-Host "== 3/6 Spark job $job"
    docker exec spark-master @submit "/opt/jobs/$job.py" 2>&1 | Tee-Object -FilePath "results\log_$job.txt" | Where-Object { "$_" -notmatch " INFO " }
    $done = (Test-Path results\run_log.jsonl) -and (Select-String -Path results\run_log.jsonl -Pattern ('"job": "' + $job.Substring(0,2)) -Quiet)
    if (-not $done) { Write-Host "Job $job failed - see results\log_$job.txt"; exit 1 }
}

Write-Host "== 4/6 Loading MongoDB"
docker exec dashboard python load_mongo.py
Write-Host "== 5/6 MongoDB example queries"
docker exec dashboard python mongo_queries.py
Write-Host "== 6/6 Done. Dashboard: http://localhost:8501   Spark: http://localhost:8080   HDFS: http://localhost:9870"
