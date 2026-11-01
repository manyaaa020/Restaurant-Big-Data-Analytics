# Spark 3.5.1 with Python; numpy/pandas are needed by Spark MLlib and toPandas()
FROM apache/spark:3.5.1-python3
USER root
RUN pip3 install --no-cache-dir numpy pandas
ENV PYSPARK_PYTHON=python3 \
    PATH=/opt/spark/bin:$PATH
WORKDIR /opt/jobs
