# Databricks notebook source
# %run ../includes/configuration

# COMMAND ----------

from pyspark.sql.functions import lit, col
from pyspark.sql import DataFrame
import datetime as dt
from pyspark.sql.types import *

# COMMAND ----------

# Write data frames to Delta tables
def write_to_delta(inputDF, tableName):
    inputDF.write \
        .mode("overwrite") \
        .format("delta") \
        .option("overwriteSchema", "true") \
        .saveAsTable(tableName)

# COMMAND ----------

# Append data frames to Delta tables
def append_to_delta(inputDF, tableName, mergeSchema = False):
  if mergeSchema:
    (inputDF.write
        .mode("append")
        .format("delta")
        .option("mergeSchema", "true")
        .saveAsTable(tableName))
  else:
    (inputDF.write
        .mode("append")
        .format("delta")
        .option("overwriteSchema", "true")
        .saveAsTable(tableName))
    

# COMMAND ----------

# Write data frames to Synapse
def write_to_synapse(inputDF, tableName):
    (inputDF.write
            .format("com.databricks.spark.sqldw")
            .mode("overwrite")
            .option("url", "jdbc:sqlserver://" + sqldwURL + ";database=" + initDB + ";user=" + userlogin + ";password=" + kvpassword + options)
            .option("forwardSparkAzureStorageCredentials", "true")
            .option("dbTable", tableName)
            .option("tempDir", tempdir)
            .save())

# COMMAND ----------

# Create DF from raw files
def read_path_bronze(rawPath: str, audit: int) -> DataFrame:
    newDF = spark.read.parquet(rawPath)
    if audit:
        print(f'DF record count: {newDF.count():,}')
    return newDF 