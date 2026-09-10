from typing import List

import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pickle
import os
from pyspark.dbutils import DBUtils
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

def read_parquet_file(containerName: str, lakeName: str, filePath: str):
    """
    Read a parquet file from the data lake
    """
    spark = SparkSession.builder.getOrCreate()
    
    fullPath="abfss://"+containerName+"@"+lakeName+filePath

    df_spark = spark.read.parquet(fullPath) 

    return df_spark

def read_csv_file(containerName: str, lakeName: str, filePath: str):
    """
    Read a csv file from the data lake
    """
    
    spark = SparkSession.builder.getOrCreate()

    fullPath="abfss://"+containerName+"@"+lakeName+filePath

    df_spark = spark.read.csv(fullPath,header=True,inferSchema=True) 

    return df_spark

def clean_column_of_strings(
    df,
    col_name: str,
    append_to_col_name: str = "_null_replaced",
    fill_string: str = "unknown",
):
    """
    Create a new column which duplicates a column of strings
    Replaces null and empty strings with a fill_string (e.g. unknown)
    """

    df = df.withColumn(
        col_name + append_to_col_name,
        F.when(
            F.col(col_name).isNull() | (F.col(col_name) == ""), fill_string
        ).otherwise(F.col(col_name)),
    )

    return df

def create_expressions_categorical_column(df, col_name: str):
    """
    Create dummy variables for a categorical column (col)
    Convert null and empty string values to string 'unknown'
    """
    # replace blank space in category with underscore
    categories = df.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
    categories_exprs = [
        F.when(F.col(col_name) == category, 1)
        .otherwise(0)
        .alias(col_name + "_" + category.replace(" ", "_"))
        for category in categories
    ]

    return categories_exprs