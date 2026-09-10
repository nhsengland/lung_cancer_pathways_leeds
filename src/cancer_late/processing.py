from pyspark.sql import functions as F
from pyspark.sql.functions import when, array, lit, col, array_union, array_distinct, size, array_except, rand, flatten
from pyspark.sql.window import Window
from pyspark.sql.types import StructType
from pyspark.sql import DataFrame as SparkDataFrame
import datetime 
import pandas as pd
import re
from src.cancer_late.utils import clean_column_of_strings, create_expressions_categorical_column
from src.cancer_late.utils import read_parquet_file, read_csv_file
from pyspark.sql.session import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

def get_latest_dict_cohort_paths(containerName,
                                 lakeName,
                                 filePath,
                                 cohort_name = "pcp_Cohort_"):
    """
    Scans the specified ADLS path for cohort parquet files, extracts their cohort dates, 
    and returns a dictionary mapping cohort date strings (YYYY-MM-DD) to their file paths.

    Args:
        containerName (str): Name of the ADLS container.
        lakeName (str): Name of the ADLS lake.
        filePath (str): Path within the container to search for cohort files.

    Returns:
        dict: Sorted dictionary where keys are cohort date strings and values are file paths.
    """
    fullPath="abfss://"+containerName+"@"+ lakeName+ filePath

    files = dbutils.fs.ls(fullPath)

    dict_cohort_paths = {}

    for f in files:
        if cohort_name in f.name:

            print("Found: ", f.path)
            df = spark.read.parquet(f.path)
            cohort_date =  df.first()["Cohort_Date"]
            cohort_date_str = datetime.datetime.strftime(cohort_date, "%Y-%m-%d")
            dict_cohort_paths[cohort_date_str] = f.path.split("abfss://"+containerName+"@"+ lakeName)[1]

    dict_cohort_paths = dict(sorted(dict_cohort_paths.items()))

    return dict_cohort_paths

def identify_and_remove_duplicate_ids(df_new_cohort, cohort_date):
    """
    Identifies and removes duplicate Patient_IDs from the given cohort DataFrame.

    Args:
        df_new_cohort (DataFrame): Spark DataFrame containing cohort data with a 'Patient_ID' column.
        cohort_date (str): The cohort date string (YYYY-MM-DD) for logging purposes.

    Returns:
        DataFrame: The input DataFrame with duplicate Patient_IDs removed.
    """
    duplicate_patient_ids = df_new_cohort.groupBy("Patient_ID").count().filter("count > 1")

    if duplicate_patient_ids.count() > 0:
        print("Duplicate IDs found in ", cohort_date)
        df_new_cohort = df_new_cohort.filter(~F.col('Patient_ID').isin([row['Patient_ID'] for row in 
        duplicate_patient_ids.select('Patient_ID').collect()]))
    else:
        print("No duplicate IDs found")

    return df_new_cohort

def process_icd10_ref(df_icd10_ref):
    """
    Processes the ICD10 reference DataFrame to keep only the latest and valid 3-character codes,
    cleans up the category descriptions, and returns a DataFrame with unique 3-character code groups.

    Args:
        df_icd10_ref (DataFrame): Input ICD10 reference Spark DataFrame.

    Returns:
        DataFrame: Processed DataFrame with unique 3-character ICD10 codes and cleaned descriptions.
    """
    # Keep latest only for ICD10 reference file
    df_icd10_ref = df_icd10_ref.filter(F.col("Effective_To").isNull())
    df_icd10_ref = df_icd10_ref.filter(F.col("Category_3_Code").isNotNull())

    # get first three characters of Alt_code -> groupby this, and find the group
    df_icd10_ref = df_icd10_ref.withColumn("Alt_Code_3_char", F.substring(F.col("Alt_Code"), 1, 3))

    windowSpec = Window.partitionBy("Alt_Code_3_char").orderBy("Alt_Code_3_char")

    df_icd10_ref_cat_3 = df_icd10_ref.withColumn("row_number", F.row_number().over(windowSpec))
    df_icd10_ref_cat_3 = df_icd10_ref_cat_3.filter(F.col("row_number") == 1).drop("row_number")

    # remove square brackets
    df_icd10_ref_cat_3 = df_icd10_ref_cat_3.withColumn("Category_3_Description", F.regexp_replace(F.col("Category_3_Description"), "Mood \\[affective\\] disorders", "Mood affective disorders"))

    # replace all empty space with underscore
    df_icd10_ref_cat_3 = df_icd10_ref_cat_3.withColumn("Category_3_Description", F.regexp_replace(F.col("Category_3_Description"), r'[\/,()\\\s-]', '_')) # replace brackets, commas, spaces, backslashes, and dashes with underscore
    df_icd10_ref_cat_3 = df_icd10_ref_cat_3.withColumn("Category_3_Description", F.regexp_replace(F.col("Category_3_Description"), r'__+', '_')) # replace two underscores with one
    df_icd10_ref_cat_3 = df_icd10_ref_cat_3.withColumn("Category_3_Description", F.regexp_replace(F.col("Category_3_Description"), r'^_|_$', '')) # remove underscore at beginning or end

    return df_icd10_ref_cat_3

def import_cancer_registry(containerName_bronze: str, lakeName: str, filePath_cancer_registration_registry: str, filePath_cancer_registration_rapid: str):
    """
    Imports and processes cancer registry and rapid cancer registry datasets, standardizes column names,
    combines them, and filters based on data source and diagnosis date.

    Args:
        containerName_bronze (str): Name of the bronze storage container.
        lakeName (str): Name of the data lake.
        filePath_cancer_registration_registry (str): Path to the cancer registration registry parquet file.
        filePath_cancer_registration_rapid (str): Path to the rapid cancer registration parquet file.

    Returns:
        DataFrame: Combined and processed cancer registry DataFrame with standardized columns.
    """
    #import cancer registry
    df_cancer_registry = read_parquet_file(containerName =containerName_bronze, 
                                           lakeName=lakeName,
                                           filePath= filePath_cancer_registration_registry)
    #update column names in cancer registry
    df_cancer_registry = df_cancer_registry\
        .withColumn("diagnosis_date", F.col("DIAGNOSISDATEBEST"))\
        .withColumn("tumour_stage", F.col("STAGE_BEST"))\
        .withColumn("tumour_site", F.col("SITE_ICD10_3CHAR"))\
        .withColumn("route", F.col("FINAL_ROUTE"))\
        .withColumn("age_at_diagnosis", F.col("AGE"))\
        .withColumn("source", F.lit("registry"))
    #import rapid cancer registry     
    df_cancer_registry_rapid = read_parquet_file(containerName =containerName_bronze, 
                                                 lakeName=lakeName,
                                                 filePath= filePath_cancer_registration_rapid)
    #update column names in rapid cancer registry      
    df_cancer_registry_rapid = df_cancer_registry_rapid\
        .withColumn("diagnosis_date", F.col("DIAGNOSISDATE"))\
        .withColumn("tumour_stage", F.col("STAGE"))\
        .withColumn("tumour_site", F.col("TUMOUR_SITE"))\
        .withColumn("route", F.col("FINAL_ROUTE"))\
        .withColumn("age_at_diagnosis", F.col("AGE"))\
        .withColumn("source", F.lit("rapid"))     
    # join cancer registry and rapid cancer registry
    df_union_cancer_registry = df_cancer_registry_rapid.select(
        "PSEUDO_NHS_NUMBER", "age_at_diagnosis","gender", "diagnosis_date", 
        "tumour_stage", "tumour_site", "route", "source", "snapshot", "dmicDateAdded")\
        .unionByName(df_cancer_registry.select(
            "PSEUDO_NHS_NUMBER", "age_at_diagnosis","gender", "diagnosis_date", 
            "tumour_stage", "tumour_site", "route", "source", "snapshot", "dmicDateAdded")
        )
    # remove null IDs
    df_union_cancer_registry = df_union_cancer_registry.filter(F.col("PSEUDO_NHS_NUMBER").isNotNull())
    
    # "Registry" data ends 2021. From 2022 onwards "rapid" is the only source available   
    df_union_cancer_registry = df_union_cancer_registry.filter((F.col("source") == "registry") | 
                                                              ((F.col("source") == "rapid") & (F.col("diagnosis_date") >= F.lit("2022-01-01"))) )

    df_union_cancer_registry = df_union_cancer_registry.dropDuplicates(["PSEUDO_NHS_NUMBER", "tumour_site", "diagnosis_date", "route", "tumour_stage"])    
    return df_union_cancer_registry

def import_cancer_deaths(containerName_bronze: str, lakeName: str, filePath_deaths: str):
    """
    Imports and processes mortality data to extract cancer-related deaths, pivots cause of death codes,
    and standardizes columns for downstream cancer registry integration.

    Args:
        containerName_bronze (str): Name of the bronze storage container.
        lakeName (str): Name of the data lake.
        filePath_deaths (str): Path to the deaths parquet file.

    Returns:
        DataFrame: Processed DataFrame of cancer-related deaths with standardized columns.
    """
    df_deaths = read_parquet_file(containerName =containerName_bronze, 
                                  lakeName=lakeName,
                                  filePath=filePath_deaths)

    # remove null ids
    df_deaths = df_deaths.filter(F.col("PSEUDO_DEC_NHS_NUMBER").isNotNull())

    df_deaths = df_deaths\
        .withColumn('year_month'                    , F.date_format('REG_DATE', 'yyyy-MM'))\
        .withColumn("date_of_death"                 , F.coalesce("REG_DATE_OF_DEATH", "REG_DATE"))\
        .withColumnRenamed("PSEUDO_DEC_NHS_NUMBER"  , "PSEUDO_NHS_NUMBER")
    
    cod_columns = ["S_UNDERLYING_COD_ICD10"] + [f"S_COD_CODE_{i}" for i in range(1, 16)]

    df_deaths_pivoted = df_deaths.selectExpr(
        "PSEUDO_NHS_NUMBER", 
        "date_of_death", 
        "stack(16, " + ", ".join([f"'{col}', {col}" for col in cod_columns]) + ") as (COD_Type, COD_Code)")

    # only keep cancer codes
    df_deaths_filtered = df_deaths_pivoted\
        .filter((F.col("COD_Code").startswith("C")) & (~F.col("COD_Code").startswith("C44")))
    # add three character tumour_site AND set the diagnosis_date to be the same as date_of_death
    df_deaths_filtered = df_deaths_filtered\
        .withColumn("tumour_site", F.expr("substring(COD_Code, 1, 3)"))\
        .withColumn("diagnosis_date", F.col("date_of_death"))\
        .withColumn("route", F.lit("cause_of_death"))\
        .withColumn("source", F.lit("mortality_data"))

    return df_deaths_filtered

def add_cancer_deaths(df_union_cancer_registry, df_deaths_filtered):
    """
    Combines cancer registry data with cancer-related deaths, ensuring consistent columns and removing null IDs.

    Args:
        df_union_cancer_registry (DataFrame): Cancer registry DataFrame.
        df_deaths_filtered (DataFrame): Cancer-related deaths DataFrame.

    Returns:
        DataFrame: Combined DataFrame of cancer registry and cancer deaths.
    """
    df_cancer_with_deaths = df_union_cancer_registry.unionByName(df_deaths_filtered.select(
                                                                              'PSEUDO_NHS_NUMBER',
                                                                              'tumour_site',
                                                                              'diagnosis_date',
                                                                              'route',
                                                                              'source'
                                                                              ),
                                                    allowMissingColumns=True
                                                    )
    df_cancer_with_deaths = df_cancer_with_deaths.filter(F.col("PSEUDO_NHS_NUMBER").isNotNull())

    return df_cancer_with_deaths

def extract_ICD10_codes_cancer(containerName_bronze: str, lakeName: str, filePath_cancer_registration_rapid: str):
    """
    Extracts unique ICD10 tumour site codes and their cancer group mappings from the rapid cancer registry,
    and appends additional mappings for completeness.

    Args:
        containerName_bronze (str): Name of the bronze storage container.
        lakeName (str): Name of the data lake.
        filePath_cancer_registration_rapid (str): Path to the rapid cancer registration parquet file.

    Returns:
        DataFrame: DataFrame mapping CANCER_GROUP to unique TUMOUR_SITE codes.
    """
    #import rapid cancer registry     
    df_cancer_registry_rapid = read_parquet_file(containerName =containerName_bronze, 
                                                 lakeName=lakeName,
                                                 filePath= filePath_cancer_registration_rapid)
    df_mapping_tumour_site_to_cancer_group = df_cancer_registry_rapid.groupBy("CANCER_GROUP").agg(F.collect_set("TUMOUR_SITE").alias("Unique_Tumour_Sites"))

    df_unique_tumour_sites = df_mapping_tumour_site_to_cancer_group.select("CANCER_GROUP", F.explode("Unique_Tumour_Sites").alias("TUMOUR_SITE"))

    new_rows = [("Lung", "C33"), ("Unknown", "C78"), ("Carinoma in situ cervix", "D06"), ("Neoplasm of uncertain or unknown behaviour", "D48"), ("Benign neoplasm of male genital organs", "D29"), ("Unknown","C39"), ("Unknown","C97")]

    df_new_rows = spark.createDataFrame(new_rows, ["CANCER_GROUP", "TUMOUR_SITE"])

    df_unique_tumour_sites = df_unique_tumour_sites.unionByName(df_new_rows)
    
    return df_unique_tumour_sites

def add_ICD10_to_cancer_registry(df_cancer_with_deaths, df_unique_tumour_sites):
    """
    Joins cancer registry and deaths data with ICD10 tumour site to cancer group mappings,
    and filters out records without a cancer group.

    Args:
        df_cancer_with_deaths (DataFrame): Combined cancer registry and deaths DataFrame.
        df_unique_tumour_sites (DataFrame): DataFrame mapping tumour sites to cancer groups.

    Returns:
        DataFrame: Cancer registry DataFrame with cancer group information.
    """
    df_cancer_ICD10 = df_cancer_with_deaths.join(df_unique_tumour_sites,
                                              on = "tumour_site",
                                              how = "left")
    df_cancer_ICD10 = df_cancer_ICD10.filter(~F.col("CANCER_GROUP").isNull())
    return df_cancer_ICD10

def get_cancer_by_group(df_cancer_ICD10):
    """
    For each patient and cancer group, identifies the earliest and most recent diagnosis dates,
    tumour stage, route, age, and gender at those dates.

    Args:
        df_cancer_ICD10 (DataFrame): Cancer registry DataFrame with cancer group information.

    Returns:
        DataFrame: Patient-level DataFrame with earliest/latest diagnosis and related attributes per cancer group.
    """
    # if we have the same ID, cancer_group and diagnosis_date, but multiple rows, take the latest one based on when it was added (this will account for cases where two routes are provided for same diagnosis
    window_spec = Window.partitionBy("PSEUDO_NHS_NUMBER", "CANCER_GROUP", "tumour_stage", "diagnosis_date").orderBy(F.desc("dmicDateAdded"))

    df_cancer_ICD10 = df_cancer_ICD10\
        .withColumn("row_number", F.row_number().over(window_spec))\
        .filter(F.col("row_number") == 1)\
        .drop("row_number") 

    #window for earliest date (order asc)
    window_spec_earliest =    Window.partitionBy("PSEUDO_NHS_NUMBER", "Cancer_Group")\
                                        .orderBy(F.col("diagnosis_date").asc())
    #window for latest date (order desc)
    window_spec_latest =      Window.partitionBy("PSEUDO_NHS_NUMBER", "Cancer_Group")\
                                        .orderBy(F.col("diagnosis_date").desc())

    df_cancer_by_group = df_cancer_ICD10\
        .withColumn("diagnosis_date_earliest"       , F.first("diagnosis_date").over(window_spec_earliest))\
        .withColumn("diagnosis_date_latest"         , F.first("diagnosis_date").over(window_spec_latest))\
        .withColumn("tumour_stage_earliest"         , F.first("tumour_stage").over(window_spec_earliest))\
        .withColumn("tumour_stage_latest"           , F.first("tumour_stage").over(window_spec_latest))\
        .withColumn("route_earliest"                , F.first("route").over(window_spec_earliest))\
        .withColumn("route_latest"                  , F.first("route").over(window_spec_latest))\
        .withColumn("age_earliest_diagnosis"        , F.first("age_at_diagnosis").over(window_spec_earliest))\
        .withColumn("gender_earliest_diagnosis"     , F.first("gender").over(window_spec_earliest))\
        .withColumn("tumour_site_earliest_diagnosis", F.first("tumour_site").over(window_spec_earliest))\
        .withColumn("source_earliest_diagnosis"     , F.first("source").over(window_spec_earliest))\
        .select(
            "PSEUDO_NHS_NUMBER",
            "Cancer_Group",
            "age_earliest_diagnosis",
            "gender_earliest_diagnosis",
            "tumour_site_earliest_diagnosis",
            "diagnosis_date_earliest",
            "diagnosis_date_latest",
            "tumour_stage_earliest",
            "tumour_stage_latest",
            "route_earliest",
            "route_latest",
            "source_earliest_diagnosis")\
        .distinct()

    df_cancer_by_group = df_cancer_by_group\
        .withColumn("YearMonth",F.date_format("diagnosis_date_earliest", "yyyy-MM"))\
        .orderBy("PSEUDO_NHS_NUMBER", "diagnosis_date_earliest")
        
    return df_cancer_by_group

def process_cancer_datasets(containerName_bronze, lakeName, filePath_cancer_registration_registry, filePath_cancer_registration_rapid, filePath_deaths):
    """
    Orchestrates the import, processing, and integration of cancer registry and mortality datasets,
    mapping tumour sites to cancer groups and summarizing by patient and group.

    Args:
        containerName_bronze (str): Name of the bronze storage container.
        lakeName (str): Name of the data lake.
        filePath_cancer_registration_registry (str): Path to the cancer registration registry parquet file.
        filePath_cancer_registration_rapid (str): Path to the rapid cancer registration parquet file.
        filePath_deaths (str): Path to the deaths parquet file.

    Returns:
        DataFrame: Patient-level DataFrame with summarized cancer group information.
    """
    df_union_cancer_registry = import_cancer_registry(
                                                containerName_bronze,
                                                lakeName,
                                                filePath_cancer_registration_registry,
                                                filePath_cancer_registration_rapid,
                                                )

    df_deaths_filtered = import_cancer_deaths(containerName_bronze,
                                              lakeName,
                                              filePath_deaths)

    df_cancer_with_deaths = add_cancer_deaths(df_union_cancer_registry, df_deaths_filtered)

    df_unique_tumour_sites = extract_ICD10_codes_cancer(containerName_bronze,
                                                 lakeName, 
                                                 filePath_cancer_registration_rapid)

    df_cancer_ICD10 = add_ICD10_to_cancer_registry(df_cancer_with_deaths, df_unique_tumour_sites)

    df_cancer_by_group = get_cancer_by_group(df_cancer_ICD10)        

    return df_cancer_by_group   

def process_111_dataset(df_111_data, df_111_reference):
    """
    Processes the 111 dataset by mapping symptom group, symptom discriminator, and disposition codes to their descriptions.

    Args:
    df_111_data (DataFrame): The raw 111 data containing the 'Dimention_4' column with codes.
    df_111_reference (DataFrame): The reference data containing mappings for SG, SD, and DX codes to their descriptions.

    Returns:
    DataFrame: The processed 111 data with mapped descriptions and cleaned column values.
    """
    # creating separate mapping files for the symptom group, symptom discriminator, and disposition
    df_111_reference_SG = df_111_reference.select(["SG", "SG Description"]).dropDuplicates()
    df_111_reference_SD = df_111_reference.select(["SD", "SD Description"]).dropDuplicates()
    df_111_reference_DX = df_111_reference.select(["DX", "DX Description"]).dropDuplicates()

    # Split the column 'dimension_4' by '╎' and create three new columns for symptom group, symptom discriminator, and disposition
    df_111_data = df_111_data.withColumn("DX", F.split(df_111_data["Dimention_4"], "¦").getItem(0)) \
                .withColumn("SG", F.split(df_111_data["Dimention_4"], "¦").getItem(1)) \
                .withColumn("SD", F.split(df_111_data["Dimention_4"], "¦").getItem(2))

    # append the string SG and SD to those columns respectively to align with how the data is stored in the mapping file 
    df_111_data = df_111_data.withColumn("SD", F.concat(F.lit("SD"), F.col("SD")))
    df_111_data = df_111_data.withColumn("SG", F.concat(F.lit("SG"), F.col("SG")))

    # join the 111 data with the mapping
    df_111_data = df_111_data.join(df_111_reference_SG, ["SG"], "left") \
                .withColumnRenamed("SG Description", "SG_Description") \
                .join(df_111_reference_SD, ["SD"], "left") \
                .withColumnRenamed("SD Description", "SD_Description") \
                .join(df_111_reference_DX, ["DX"], "left") \
                .withColumnRenamed("DX Description", "DX_Description")

    columns_to_replace = ["Dimention_1", "SG_Description", "SD_Description", "DX_Description"]
    for col in columns_to_replace:
        df_111_data = df_111_data.withColumn(col, F.regexp_replace(col, r'[\/,()\\\s]', '_')) # replace brackets, commas, spaces, and backslashes with underscore
        df_111_data = df_111_data.withColumn(col, F.regexp_replace(col, r'__+', '_')) # replace two underscores with one
        df_111_data = df_111_data.withColumn(col, F.regexp_replace(col, r'^_|_$', '')) # remove underscore at beginning or end
    
    return df_111_data

def process_gp_data_emis_s1(df_pcp_emis, df_pcp_s1):
    """
    Processes GP data from EMIS and S1 systems, standardizes column names, and combines the datasets.

    Args:
        df_pcp_emis (DataFrame): DataFrame containing EMIS GP data.
        df_pcp_s1 (DataFrame): DataFrame containing S1 GP data.

    Returns:
        DataFrame: Combined DataFrame with standardized columns and deduplicated records.
    """
    # pcp_emis create new column Patient_ID (Patient_Pseudonym) , Event_Date (EffectiveDateTime), NumericValue (NumericValue), NumericUnits (NumericUnits), SnomedCode (SnomedCTCode)
    pcp_emis_select = df_pcp_emis.select("Patient_Pseudonym", "EffectiveDateTime", "NumericValue", "NumericUnits", "SnomedCTCode")
    pcp_emis_select = pcp_emis_select.withColumnRenamed("Patient_Pseudonym", "Patient_ID").withColumnRenamed("EffectiveDateTime", "Attendance_Date").withColumnRenamed("SnomedCTCode", "SnomedCode")
    pcp_emis_select = pcp_emis_select.withColumn("source", F.lit("emis"))

    # pcp_s1 create new column Patient_ID (Patient_Pseudonym) , Event_Date (DateEvent), NumericValue (NumericValue), NumericUnits (NumericUnit), SnomedCode (SNOMEDCode)
    pcp_s1_select = df_pcp_s1.select("Patient_Pseudonym", "DateEvent", "NumericValue", "NumericUnit", "SNOMEDCode")
    pcp_s1_select = pcp_s1_select.withColumnRenamed("Patient_Pseudonym", "Patient_ID").withColumnRenamed("DateEvent", "Attendance_Date").withColumnRenamed("SNOMEDCode", "SnomedCode").withColumnRenamed("NumericUnit", "NumericUnits")
    pcp_s1_select = pcp_s1_select.withColumn("source", F.lit("s1"))

    df_gp_events_emis_s1 = pcp_emis_select.unionByName(pcp_s1_select)

    df_gp_events_emis_s1 = df_gp_events_emis_s1.withColumn("Attendance_Date_date", F.to_date("Attendance_Date"))

    df_gp_events_emis_s1 = df_gp_events_emis_s1.dropDuplicates(subset=["Patient_ID", "Attendance_Date_date", "SnomedCode"])

    return df_gp_events_emis_s1