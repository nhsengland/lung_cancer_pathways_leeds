# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 4. Construct Flags for Control Cohort
# MAGIC This notebook produces tables containing lung cancer flags appearing in the patient records of the control cohort created in Notebook 3. The purpose of this is to determine how informative these flags are for early detection of lung cancer. For example, by comparing the control cohort to the lung cancer cohort we can investigate how many patients which report shortness of breath actually go on to develop lung cancer in the next year.

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import src.cancer_late.config  as config
import src.cancer_late.config_pathways  as config_pathways
from src.cancer_late.utils import read_parquet_file, read_csv_file
import src.cancer_late.patient_pathways_utils as patient_pathways_utils
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew, kurtosis, anderson_ksamp, kruskal
import pandas as pd
from operator import add
from functools import reduce
import plotly.express as px
import plotly.graph_objects as go
from src.cancer_late import processing
from pyspark.sql.window import Window

# COMMAND ----------

cancer_site = "Lung"
control_site = "Lung_control"
version = config_pathways.cancer_site_mappings[cancer_site]["run_version"]
history_days = 365

# COMMAND ----------

# MAGIC %md
# MAGIC # Import datasets

# COMMAND ----------

df_latest_cohort = read_parquet_file(containerName =config.containerName_platinum, 
                                     lakeName=config.lakeName,
                                     filePath= f"")

df_control_site = read_parquet_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= f"")

df_all_activity = read_parquet_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= f"")

df_111_mapping = read_csv_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= config_pathways.cancer_site_mappings[cancer_site]["flags_111"])

df_ecds_mapping = read_csv_file(containerName =config.containerName_platinum, 
                                lakeName=config.lakeName,
                                filePath= config_pathways.cancer_site_mappings[cancer_site]["flags_ecds"])

# COMMAND ----------

paths_amber_flag_gp = config_pathways.cancer_site_mappings[cancer_site]["amber_flag_gp"]

if len(paths_amber_flag_gp)>0:

    i=0
    for single_symptom in paths_amber_flag_gp:

        df = read_csv_file(containerName =config.containerName_platinum, 
                           lakeName=config.lakeName,
                           filePath= single_symptom["path"])
        
        if "concept_id" in df.columns:
            df=df.withColumnRenamed("concept_id", "code")
            df = df.select(["code", "term"])

        elif "medcodeid" in df.columns:
            df=df.withColumnRenamed("medcodeid", "code")
            df = df.select(["code", "term"])

        df = df.withColumn("symptom_name",F.lit(single_symptom["name"]))

        if i==0:
            df_gp_amber_flags_ref = df

        else:
            df_gp_amber_flags_ref = df_gp_amber_flags_ref.unionByName(df)            

        print(f"GP Amber Flags, added: {single_symptom['name']}")

        i=i+1

else:
    # create empty pyspark dataframe with columns code, term, symptom_name
    
    schema = StructType([
        StructField("code", LongType(), True),
        StructField("term", StringType(), True),
        StructField("symptom_name", StringType(), True)
    ])

    df_gp_amber_flags_ref = spark.createDataFrame([], schema)

df_gp_amber_flags_ref = df_gp_amber_flags_ref.drop_duplicates(subset=["code"])

# COMMAND ----------

paths_red_flag_gp = config_pathways.cancer_site_mappings[cancer_site]["red_flag_gp"]

if len(paths_red_flag_gp)>0:

    i=0
    for single_symptom in paths_red_flag_gp:

        df = read_csv_file(containerName =config.containerName_platinum, 
                                                  lakeName=config.lakeName,
                                                  filePath= single_symptom["path"])

        if "concept_id" in df.columns:
            df=df.withColumnRenamed("concept_id", "code")
            df = df.select(["code", "term"])
            
        elif "medcodeid" in df.columns:
            df=df.withColumnRenamed("medcodeid", "code")
            df = df.select(["code", "term"])

        df = df.withColumn("symptom_name",F.lit(single_symptom["name"]))

        if i==0:
            df_gp_red_flags_ref = df

        else:
            df_gp_red_flags_ref = df_gp_red_flags_ref.unionByName(df)            

        print(f"GP Red Flags, added: {single_symptom['name']}")

        i=i+1

else:
    # create empty pyspark dataframe with columns code, term, symptom_name
    
    schema = StructType([
        StructField("code", LongType(), True),
        StructField("term", StringType(), True),
        StructField("symptom_name", StringType(), True)
    ])

    df_gp_red_flags_ref = spark.createDataFrame([], schema)

df_gp_red_flags_ref = df_gp_red_flags_ref.drop_duplicates(subset=["code"])


# COMMAND ----------

# screening breakdown

schema = StructType([
    StructField("code", LongType(), True),
    StructField("term", StringType(), True),
    StructField("grouping", StringType(), True)
])

df_screening_mapping = spark.createDataFrame([], schema)

if config_pathways.cancer_site_mappings[cancer_site]["screening"]:

    for screening_status in ["screening_attended", "screening_did_not_attend"]:

        df_screening = read_csv_file(containerName =config.containerName_platinum, 
                                            lakeName=config.lakeName,
                                            filePath= config_pathways.cancer_site_mappings[cancer_site][screening_status])
        
        df_screening = df_screening.withColumn("grouping", F.lit(screening_status)) 

        df_screening_mapping = df_screening_mapping.unionByName(df_screening)

# COMMAND ----------

paths_referrals = config_pathways.cancer_site_mappings[cancer_site]["cancer_referral"]

if len(paths_referrals)>0:

    i=0
    for single_path in paths_referrals:

        df = read_csv_file(containerName =config.containerName_platinum, 
                                                  lakeName=config.lakeName,
                                                  filePath= single_path)

        if i==0:
            df_referral_mapping = df

        else:
            df_referral_mapping = df_referral_mapping.unionByName(df)           

        print(f"Cancer referrals, added: {single_path}")

        i=i+1

else:
    
    schema = StructType([
        StructField("code", LongType(), True),
        StructField("term", StringType(), True)
    ])

    df_referral_mapping = spark.createDataFrame([], schema)

# COMMAND ----------

if cancer_site == "Lung":
    df_mapping_chest_xray = read_csv_file(containerName =config.containerName_platinum, 
                                        lakeName=config.lakeName,
                                        filePath= config_pathways.cancer_site_mappings[cancer_site]["chest_xray_snomed"])

    medication_table_paths = config_pathways.cancer_site_mappings[cancer_site]["medication_code_tables"]
    medication_dfs = {}
    for medication_name, medication_table_path in medication_table_paths.items():
        medication_df = read_csv_file(containerName =config.containerName_platinum, 
                                            lakeName=config.lakeName,
                                            filePath= medication_table_path)
        medication_dfs[medication_name] = medication_df


    df_copd_icd10 = read_csv_file(containerName =config.containerName_platinum, 
                                        lakeName=config.lakeName,
                                        filePath= config_pathways.cancer_site_mappings[cancer_site]["copd_icd10"])
    
    df_copd_snomed = read_csv_file(containerName =config.containerName_platinum, 
                                        lakeName=config.lakeName,
                                        filePath= config_pathways.cancer_site_mappings[cancer_site]["copd_snomed"])
    
    df_smi_snomed = read_csv_file(containerName =config.containerName_platinum, 
                                        lakeName=config.lakeName,
                                        filePath= config_pathways.smi_snomed)

    df_smi_icd10 = read_csv_file(containerName =config.containerName_platinum, 
                                        lakeName=config.lakeName,
                                        filePath= config_pathways.smi_icd10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Activity table

# COMMAND ----------

print("Size of activity table before filtering to events prior to diagnosis: ", df_all_activity.count())

# only consider activities at least one day before diagnosis
df_all_activity_before_diagnosis = df_all_activity.filter(F.col("days_between_activity_diagnosis")>=0)

print("Size of activity table after filtering to events prior to diagnosis: ", df_all_activity_before_diagnosis.count())

# COMMAND ----------

# check for duplicate rows 
display(df_all_activity_before_diagnosis.groupBy(df_all_activity_before_diagnosis.columns).count().filter(F.col("count") > 1))

# COMMAND ----------

# MAGIC %md
# MAGIC # Acute flags

# COMMAND ----------

# MAGIC %md
# MAGIC ## icd10 red flag

# COMMAND ----------

if config_pathways.cancer_site_mappings[cancer_site]["icd10_red_flag_codes"]:
    df_icd10_codes = read_csv_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= config_pathways.cancer_site_mappings[cancer_site]["icd10_red_flag_codes"])
       
    df_all_activity_before_diagnosis_acute_sus_events = df_all_activity_before_diagnosis.filter(F.col("Diagnosis_type")=="ICD10")
    
    # Use OR condition on full Event code OR 3‑char prefix
    df_acute_red_flag = df_all_activity_before_diagnosis_acute_sus_events.join(df_icd10_codes.select(["ICD10", "NICE_matching"]),
                                                                                        (
                                                                                            (df_all_activity_before_diagnosis_acute_sus_events.Event_Code == df_icd10_codes.ICD10)
                                                                                            |
                                                                                            (df_all_activity_before_diagnosis_acute_sus_events.Diagnosis_SUS_icd10_3_char == df_icd10_codes.ICD10)
                                                                                            ),
                                                                                        'inner')

    df_acute_red_flag_patients_summary = patient_pathways_utils.create_flags_summary(df_mapped_events=df_acute_red_flag,
                                                               history_days=history_days,
                                                               dataset="acute",
                                                               col_name="NICE_matching",
                                                               name_of_event="acute_red_flag")
   


display(df_acute_red_flag_patients_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC ## icd10 amber flag

# COMMAND ----------

if config_pathways.cancer_site_mappings[cancer_site]["icd10_amber_flag_codes"]:
    df_icd10_codes = read_csv_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= config_pathways.cancer_site_mappings[cancer_site]["icd10_amber_flag_codes"])
    
    df_icd10_codes = df_icd10_codes.withColumnRenamed("Description", "acute_diagnosis")

    df_all_activity_before_diagnosis_acute_sus_events = df_all_activity_before_diagnosis.filter(F.col("Diagnosis_type")=="ICD10")

    # Use OR condition on full Event code OR 3‑char prefix

    df_acute_amber_flag = df_all_activity_before_diagnosis_acute_sus_events.join(df_icd10_codes.select(["ICD10", "NICE_matching"]),
                                                                                            (
                                                                                                (df_all_activity_before_diagnosis_acute_sus_events.Event_Code == df_icd10_codes.ICD10)
                                                                                                |
                                                                                                (df_all_activity_before_diagnosis_acute_sus_events.Diagnosis_SUS_icd10_3_char == df_icd10_codes.ICD10)
                                                                                                ),
                                                                                            'inner')


    df_acute_amber_flag_patients_summary = patient_pathways_utils.create_flags_summary(df_mapped_events=df_acute_amber_flag,
                                                               history_days=history_days,
                                                               dataset="acute",
                                                               col_name="NICE_matching",
                                                               name_of_event="acute_amber_flag")
   


display(df_acute_amber_flag_patients_summary)
        

# COMMAND ----------

# MAGIC %md
# MAGIC # ECDS flags

# COMMAND ----------

# MAGIC %md
# MAGIC ## red flags

# COMMAND ----------

if config_pathways.cancer_site_mappings[cancer_site]["flags_ecds"]:

    df_ecds_mapping_red_flag = df_ecds_mapping.filter(F.col("grouping")=="red_flag").withColumnRenamed("SNOMED_UK_Preferred_Term", "ecds_chief_complaint")

    df_all_activity_before_diagnosis_ecds_events = df_all_activity_before_diagnosis.filter(F.col("dataset") == 'ECDS')

    df_ecds_red_flags = df_all_activity_before_diagnosis_ecds_events.join(df_ecds_mapping_red_flag,
                                                                          df_all_activity_before_diagnosis_ecds_events.Emergency_Care_Chief_Complaint_Snomed_CT == df_ecds_mapping_red_flag.SNOMED_Code, 'inner')
    

    df_ecds_red_flags_patient_summary = patient_pathways_utils.create_flags_summary(df_mapped_events=df_ecds_red_flags,
                                                               history_days=history_days,
                                                               dataset="ecds",
                                                               col_name="NICE_matching",
                                                               name_of_event="ecds_red_flag")
   


display(df_ecds_red_flags_patient_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## amber flags

# COMMAND ----------

if config_pathways.cancer_site_mappings[cancer_site]["flags_ecds"]:

    df_ecds_mapping_amber_flag = df_ecds_mapping.filter(F.col("grouping")=="amber_flag").withColumnRenamed("SNOMED_UK_Preferred_Term", "ecds_chief_complaint")

    df_all_activity_before_diagnosis_ecds_events = df_all_activity_before_diagnosis.filter(F.col("dataset") == 'ECDS')

    df_ecds_amber_flags = df_all_activity_before_diagnosis_ecds_events.join(df_ecds_mapping_amber_flag,
                                                                          df_all_activity_before_diagnosis_ecds_events.Emergency_Care_Chief_Complaint_Snomed_CT == df_ecds_mapping_amber_flag.SNOMED_Code, 'inner')
    

    df_ecds_amber_flags_patient_summary = patient_pathways_utils.create_flags_summary(df_mapped_events=df_ecds_amber_flags,
                                                               history_days=history_days,
                                                               dataset="ecds",
                                                               col_name="NICE_matching",
                                                               name_of_event="ecds_amber_flag")
   

display(df_ecds_amber_flags_patient_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # 111 flags

# COMMAND ----------

# MAGIC %md
# MAGIC ## red flags

# COMMAND ----------

if config_pathways.cancer_site_mappings[cancer_site]["flags_111"]:
    
    df_111_red_flags = df_all_activity_before_diagnosis.join(df_111_mapping.filter(F.col("grouping")=="red_flag"), df_all_activity_before_diagnosis.SG_Description == df_111_mapping.symptom_111, 'inner')

    df_111_red_flags_patient_summary = patient_pathways_utils.create_flags_summary(df_mapped_events=df_111_red_flags,
                                                               history_days=history_days,
                                                               dataset="111",
                                                               col_name="NICE_matching",
                                                               name_of_event="111_red_flag")
   


display(df_111_red_flags_patient_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC ## amber flags

# COMMAND ----------

if config_pathways.cancer_site_mappings[cancer_site]["flags_111"]:
    
    df_111_amber_flags = df_all_activity_before_diagnosis.join(df_111_mapping.filter(F.col("grouping")=="amber_flag"), df_all_activity_before_diagnosis.SG_Description == df_111_mapping.symptom_111, 'inner')

    df_111_amber_flags_patient_summary = patient_pathways_utils.create_flags_summary(df_mapped_events=df_111_amber_flags,
                                                               history_days=history_days,
                                                               dataset="111",
                                                               col_name="NICE_matching",
                                                               name_of_event="111_amber_flag")

else:
    schema = StructType([
    StructField("Patient_ID", StringType(), True),
])

    df_111_amber_flags_patient_summary = spark.createDataFrame([], schema = schema)
   
display(df_111_amber_flags_patient_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # GP Flags

# COMMAND ----------

df_all_activity_before_diagnosis_gp_events = df_all_activity_before_diagnosis.filter(F.col("dataset") == 'gp_events')

df_gp_amber_flags_ref = df_gp_amber_flags_ref.withColumn("grouping", F.lit("amber_flag"))
df_gp_red_flags_ref = df_gp_red_flags_ref.withColumn("grouping", F.lit("red_flag"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Red flags

# COMMAND ----------

df_gp_red_flags = df_all_activity_before_diagnosis_gp_events.join(df_gp_red_flags_ref.drop("term"),
                                                                  df_all_activity_before_diagnosis_gp_events.Concept_ID == df_gp_red_flags_ref.code, 'inner')


# Add thrombocytosis via high platelet count
df_patients_with_platelet_count = df_all_activity_before_diagnosis.filter(F.col("description")=="Platelet count")

df_patients_with_platelet_count = df_patients_with_platelet_count.withColumn(
    "NumericValue_clean",
    F.regexp_replace(F.col("NumericValue"), "[^0-9.]", "").cast("double")
)

df_patients_with_platelet_count = df_patients_with_platelet_count.withColumn("flag_high_platelet_count", 
                                                                             F.when(F.col("NumericValue_clean")>=400,1).otherwise(0)
                                                                             )
                                                                             
df_patients_with_platelet_count = df_patients_with_platelet_count.withColumn("symptom_name", F.lit("thrombocytosis"))
df_patients_with_platelet_count = df_patients_with_platelet_count.withColumn("grouping", F.lit("red_flag"))

df_gp_red_flags = df_gp_red_flags.unionByName(df_patients_with_platelet_count.filter(F.col("flag_high_platelet_count")==1), allowMissingColumns = True)

df_gp_red_flags_patient_summary = patient_pathways_utils.create_flags_summary(df_mapped_events=df_gp_red_flags,
                                                               history_days=history_days,
                                                               dataset="gp",
                                                               col_name="symptom_name",
                                                               name_of_event="gp_red_flag")

display(df_gp_red_flags_patient_summary)




# COMMAND ----------

# MAGIC %md
# MAGIC ## Amber Flags

# COMMAND ----------

df_gp_amber_flags = df_all_activity_before_diagnosis_gp_events.join(df_gp_amber_flags_ref.drop("term"),
                                                                    df_all_activity_before_diagnosis_gp_events.Concept_ID == df_gp_amber_flags_ref.code,
                                                                    'inner')

df_gp_amber_flags_patient_summary = patient_pathways_utils.create_flags_summary(df_mapped_events=df_gp_amber_flags,
                                                               history_days=history_days,
                                                               dataset="gp",
                                                               col_name="symptom_name",
                                                               name_of_event="gp_amber_flag")


display(df_gp_amber_flags_patient_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC # Screening

# COMMAND ----------

# MAGIC %md
# MAGIC ## SNOMED

# COMMAND ----------

df_screening_snomed = df_all_activity_before_diagnosis.join(df_screening_mapping.select(["code", "grouping"]),
                                                     df_all_activity_before_diagnosis.Concept_ID == df_screening_mapping.code,
                                                     'inner')
df_screening_snomed = df_screening_snomed.filter(F.col("days_between_activity_diagnosis")<=history_days)

df_screening_snomed = df_screening_snomed.select(["Patient_ID", "code", "grouping", "date", "days_between_activity_diagnosis","description"])


# COMMAND ----------

# MAGIC %md
# MAGIC ## icd10

# COMMAND ----------

df_screening_attended_icd10=df_all_activity_before_diagnosis.filter((F.col("Diagnosis_type")=="ICD10") 
                                                           & F.col("Event_Code").startswith(config_pathways.cancer_site_mappings[cancer_site]["screening_icd10_attended"]))

df_screening_attended_icd10=df_screening_attended_icd10.withColumn("grouping", F.lit("screening_attended"))

df_screening_abnormal_icd10=df_all_activity_before_diagnosis.filter((F.col("Diagnosis_type")=="ICD10") 
                                                           & F.col("Event_Code").startswith(config_pathways.cancer_site_mappings[cancer_site]["screening_icd10_abnormal"]))

df_screening_abnormal_icd10=df_screening_abnormal_icd10.withColumn("grouping", F.lit("screening_abnormal"))

df_screening_icd10 = df_screening_attended_icd10.unionByName(df_screening_abnormal_icd10, allowMissingColumns=True)

df_screening_icd10=df_screening_icd10.select(["Patient_ID", "Event_Code", "grouping", "date", "days_between_activity_diagnosis","description"])

# COMMAND ----------

# MAGIC %md
# MAGIC ##combine

# COMMAND ----------

display(df_screening_snomed)

# COMMAND ----------

df_screening = df_screening_snomed.unionByName(df_screening_icd10, allowMissingColumns=True)

df_first_screening_dates = (
    df_screening
    .filter(F.col("grouping").isin(["screening_attended", "screening_abnormal", "screening_normal"]))
    .groupby("Patient_ID")
    .agg(
        F.min(F.when(F.col("grouping") .isin(["screening_attended", "screening_abnormal", "screening_normal"]) , F.col("date"))).alias("first_screening_date"),

        F.min(F.when(F.col("grouping") == "screening_abnormal", F.col("date"))).alias("first_abnormal_screening_date"),
        
        F.min(F.when(F.col("grouping") == "screening_normal", F.col("date"))).alias("first_normal_screening_date"),
    )
    .filter(
        (F.col("first_screening_date").isNotNull()) |
        (F.col("first_abnormal_screening_date").isNotNull()) |
        (F.col("first_normal_screening_date").isNotNull())
    )
)

df_screening_result = df_screening.groupby("Patient_ID").pivot("grouping").count()

# fill with null if column does not exist
for col in ["screening_abnormal", "screening_attended", "screening_abnormal", "screening_did_not_attend"]:
    if col not in df_screening_result.columns:
        df_screening_result = df_screening_result.withColumn(col, F.lit(None).cast(IntegerType())) 


df_screening_result = df_screening_result.withColumn("flag_abnormal_screening", F.when((F.col("screening_abnormal")>0), 1).otherwise(0))

#df_screening_result = df_screening_result.withColumn("flag_normal_screening", F.when((F.col("screening_normal")>0), 1).otherwise(0))
df_screening_result = df_screening_result.withColumn("flag_attended_screening", F.when((F.col("screening_attended")>0) | 
                                                                                  (F.col("screening_abnormal")>0), 1).otherwise(0))

df_screening_result = df_screening_result.withColumn("flag_did_not_attend_screening", F.when((F.col("screening_did_not_attend")>0) & 
                                                                                             (F.col("flag_attended_screening")==0), 1).otherwise(0))


df_screening_result_patient_summary = df_screening_result.join(df_first_screening_dates, on = "Patient_ID", how="left")

# breakdown screening codes

df_screening.groupby("grouping").count().display()  

# COMMAND ----------

# MAGIC %md
# MAGIC # Referral

# COMMAND ----------

df_referral = df_all_activity_before_diagnosis.join(
    df_referral_mapping, 
    df_all_activity_before_diagnosis.Concept_ID == df_referral_mapping.code,
    'inner'
).drop(df_referral_mapping.term) 

df_referral_summary = patient_pathways_utils.count_and_first_date_of_event(df_event_table=df_referral,
                                                                           history_days = history_days,
                                                                           name_of_event="gp_referral_suspected")

# COMMAND ----------

# MAGIC %md
# MAGIC # Chest X-ray

# COMMAND ----------

df_chest_xray_snomed = df_all_activity_before_diagnosis.join(df_mapping_chest_xray.select(["code"]),
                                                     df_all_activity_before_diagnosis.Concept_ID == df_mapping_chest_xray.code,
                                                     'inner')
df_chest_xray_snomed = df_chest_xray_snomed.filter(F.col("days_between_activity_diagnosis")<=history_days)

df_chest_xray_snomed = df_chest_xray_snomed.withColumn("grouping", F.lit("chest_xray"))

df_chest_xray_snomed = df_chest_xray_snomed.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "grouping"])

# COMMAND ----------

df_chest_xray_icd10=df_all_activity_before_diagnosis.filter((F.col("Diagnosis_type")=="ICD10") 
                                                           & F.col("Event_Code").startswith(config_pathways.cancer_site_mappings[cancer_site]["abnormal_xray_icd10"]))

df_chest_xray_icd10=df_chest_xray_icd10.withColumn("grouping", F.lit("chest_xray"))

df_chest_xray_icd10 = df_chest_xray_icd10.filter(F.col("days_between_activity_diagnosis")<=history_days)

df_chest_xray_icd10 = df_chest_xray_icd10.withColumn("grouping", F.lit("chest_xray"))

df_chest_xray_icd10 = df_chest_xray_icd10.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "grouping"])

# COMMAND ----------

df_chest_xray_procedure = df_all_activity_before_diagnosis.filter((F.col("Procedure_type")=="OPCS") 
                                                           & F.col("Event_Code").startswith("U07"))


df_chest_xray_procedure = df_chest_xray_procedure.withColumn("grouping", F.lit("chest_xray"))

df_chest_xray_procedure = df_chest_xray_procedure.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "grouping"])

# COMMAND ----------

df_chest_xray_combined = df_chest_xray_snomed.unionByName(df_chest_xray_icd10)
df_chest_xray_combined = df_chest_xray_combined.unionByName(df_chest_xray_procedure)
df_chest_xray_combined = df_chest_xray_combined.dropDuplicates(subset = ["Patient_ID", "date"])

# COMMAND ----------

df_chest_xray_summary = patient_pathways_utils.count_and_first_date_of_event(df_event_table=df_chest_xray_combined,
                                                                           history_days = history_days,
                                                                           name_of_event="chest_xray")

# COMMAND ----------

# find most recent chest x-ray

df_chest_xray_latest_date = df_chest_xray_combined.groupBy("Patient_ID").agg(
    F.max(F.col("date")).alias("date_last_reported_chest_xray_in_last_" + str(history_days) + "_days")
)

df_chest_xray_summary = df_chest_xray_summary.join(df_chest_xray_latest_date, on = "Patient_ID", how="inner")

# COMMAND ----------

display(df_chest_xray_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # COPD flag

# COMMAND ----------

copd_years_history = config_pathways.cancer_site_mappings[cancer_site]["copd_look_back_years"]

df_copd_snomed_activity = df_all_activity_before_diagnosis.join(df_copd_snomed.select(["code"]),
                                                     df_all_activity_before_diagnosis.Concept_ID == df_copd_snomed.code,
                                                     'inner')

df_copd_snomed_activity = df_copd_snomed_activity.withColumn("grouping", F.lit("copd_diagnosis"))

df_copd_snomed_activity = df_copd_snomed_activity.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "grouping"])

df_copd_icd10_activity=df_all_activity_before_diagnosis.join(df_copd_icd10.select(["code"]),
                                                     df_all_activity_before_diagnosis.Event_Code.startswith(df_copd_icd10.code),
                                                     'inner')

df_copd_icd10_activity=df_copd_icd10_activity.withColumn("grouping", F.lit("copd_diagnosis"))

df_copd_icd10_activity = df_copd_icd10_activity.withColumn("grouping", F.lit("copd_diagnosis"))

df_copd_icd10_activity = df_copd_icd10_activity.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "grouping"])

df_copd_combined = df_copd_snomed_activity.unionByName(df_copd_icd10_activity)
df_copd_combined = df_copd_combined.dropDuplicates(subset = ["Patient_ID", "date"])

df_copd_summary = patient_pathways_utils.count_and_first_date_of_event(df_event_table=df_copd_combined,
                                                                           history_days = history_days*copd_years_history, # COPD diagnosis in last copd_years_history years
                                                                           name_of_event="copd_diagnosis")


# Binary flag
df_copd_summary = df_copd_summary.withColumn(f"flag_copd_diagnosis",
                                               F.when((F.col(f"number_of_times_copd_diagnosis_in_last_{str(history_days*copd_years_history)}_days")>0), 1).otherwise(0))


df_copd_summary = df_copd_summary.select(["Patient_ID", "flag_copd_diagnosis"])

display(df_copd_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC # Severe Mental Illness Flag

# COMMAND ----------

years_history = config_pathways.smi_look_back_years

df_smi_snomed_activity = df_all_activity_before_diagnosis.join(df_smi_snomed.select(["code"]),
                                                     df_all_activity_before_diagnosis.Concept_ID == df_smi_snomed.code,
                                                     'inner')

df_smi_snomed_activity = df_smi_snomed_activity.withColumn("grouping", F.lit("smi_diagnosis"))

df_smi_snomed_activity = df_smi_snomed_activity.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "grouping"])

df_smi_icd10_activity=df_all_activity_before_diagnosis.join(df_smi_icd10.select(["code"]),
                                                     df_all_activity_before_diagnosis.Event_Code.startswith(df_smi_icd10.code),
                                                     'inner')

df_smi_icd10_activity=df_smi_icd10_activity.withColumn("grouping", F.lit("smi_diagnosis"))

df_smi_icd10_activity = df_smi_icd10_activity.withColumn("grouping", F.lit("smi_diagnosis"))

df_smi_icd10_activity = df_smi_icd10_activity.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "grouping"])

df_smi_combined = df_smi_snomed_activity.unionByName(df_smi_icd10_activity)
df_smi_combined = df_smi_combined.dropDuplicates(subset = ["Patient_ID", "date"])

df_smi_summary = patient_pathways_utils.count_and_first_date_of_event(df_event_table=df_smi_combined,
                                                                      history_days = history_days*years_history, # smi diagnosis in last years_history
                                                                      name_of_event="smi_diagnosis")

# Binary flag
df_smi_summary = df_smi_summary.withColumn(f"flag_smi_diagnosis",
                                               F.when((F.col(f"number_of_times_smi_diagnosis_in_last_{str(history_days*years_history)}_days")>0), 1).otherwise(0))


df_smi_summary = df_smi_summary.select(["Patient_ID", "flag_smi_diagnosis"])

display(df_smi_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC # GP Meds Flags

# COMMAND ----------

medication_summary_dfs = []

for medication_name, medication_df in medication_dfs.items():    
    df_medication_snomed = df_all_activity_before_diagnosis.join(medication_df.select(["code"]),
                                                         df_all_activity_before_diagnosis.Concept_ID == medication_df.code,
                                                         'inner')
    df_medication_snomed = df_medication_snomed.filter(F.col("days_between_activity_diagnosis")<=history_days)
    
    df_medication_snomed = df_medication_snomed.withColumn("grouping", F.lit(medication_name))
    
    df_medication_snomed = df_medication_snomed.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "grouping"])
    df_medication_summary = patient_pathways_utils.count_and_first_date_of_event(df_event_table=df_medication_snomed,
                                                                           history_days = history_days,
                                                                           name_of_event=medication_name)
    
    df_medication_latest_date = df_medication_snomed.groupBy("Patient_ID").agg(
    F.max(F.col("date")).alias(f"date_last_reported_{medication_name}_in_last_" + str(history_days) + "_days")
    )

    df_medication_summary = df_medication_summary.join(df_medication_latest_date, on = "Patient_ID", how="inner")
    
    
    # Binary flag
    df_medication_summary = df_medication_summary.withColumn(f"flag_{medication_name}_in_last_{str(history_days)}_days",
                                               F.when((F.col(f"number_of_times_{medication_name}_in_last_{str(history_days)}_days")>0), 1).otherwise(0))
    medication_summary_dfs.append(df_medication_summary)

# COMMAND ----------

display(medication_summary_dfs[0])

# COMMAND ----------

# MAGIC %md
# MAGIC # Patient summary of journey

# COMMAND ----------

# MAGIC %md
# MAGIC - join cancer site with flags
# MAGIC - fill nulls with appropriate value
# MAGIC - construct metrics

# COMMAND ----------

df_patient_flags = df_control_site.join(df_latest_cohort.drop("diagnosis_date_earliest"),
                                       on = "Patient_ID",
                                       how="inner")

df_patient_flags = df_patient_flags.join(df_copd_summary, on = "Patient_ID", how = "left").fillna(0, subset = "flag_copd_diagnosis")

# update COPD flag
df_patient_flags = df_patient_flags.withColumn("LTC_COPD",
                            F.when((F.col("LTC_COPD")==1) | (F.col("flag_copd_diagnosis")==1),1).otherwise(0))

df_patient_flags = df_patient_flags.join(df_smi_summary, on = "Patient_ID", how = "left").fillna(0, subset = "flag_smi_diagnosis")

print(df_patient_flags.count())

# COMMAND ----------

if "ecds" in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]:
    df_patient_flags = df_patient_flags.join(df_ecds_red_flags_patient_summary,
                                        on = "Patient_ID",
                                        how="left")

    df_patient_flags = df_patient_flags.join(df_ecds_amber_flags_patient_summary,
                                        on = "Patient_ID",
                                        how="left")

if "111" in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]:
    df_patient_flags = df_patient_flags.join(df_111_red_flags_patient_summary,
                                        on = "Patient_ID",
                                        how="left")

    df_patient_flags = df_patient_flags.join(df_111_amber_flags_patient_summary,
                                        on = "Patient_ID",
                                        how="left")

if "gp" in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]:
    df_patient_flags = df_patient_flags.join(df_gp_red_flags_patient_summary,
                                        on = "Patient_ID",
                                        how="left")

    df_patient_flags = df_patient_flags.join(df_gp_amber_flags_patient_summary,
                                            on = "Patient_ID",
                                            how="left")

if "acute" in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]:

    df_patient_flags = df_patient_flags.join(df_acute_red_flag_patients_summary,
                                        on = "Patient_ID",
                                        how="left")
    
    df_patient_flags = df_patient_flags.join(df_acute_amber_flag_patients_summary,
                                        on = "Patient_ID",
                                        how="left")


if config_pathways.cancer_site_mappings[cancer_site]["screening"] == True:
    df_patient_flags = df_patient_flags.join(df_screening_result_patient_summary,
                                            on = "Patient_ID",
                                            how="left")

    df_patient_flags = df_patient_flags.join(df_referral_summary,
                                            on = "Patient_ID",
                                            how="left")

    df_patient_flags = df_patient_flags.join(df_chest_xray_summary,
                                            on = "Patient_ID",
                                            how="left")

if config_pathways.cancer_site_mappings[cancer_site]["medication"] == True:
    for medication_summary_df in medication_summary_dfs:
        df_patient_flags = df_patient_flags.join(medication_summary_df,
                                                 on = "Patient_ID",
                                                 how="left")

# COMMAND ----------

cols_fill_zero = []
for col in df_patient_flags.columns:
    if "number_of_times" in col:
        cols_fill_zero.append(col)

cols_fill_zero.append("flag_attended_screening")



# COMMAND ----------

# if tumour stage is 1 or 2, put early, if 3 or 4, put late, otherwise put unknown

df_patient_flags = df_patient_flags.withColumn(
    "tumour_stage_group",
    F.lit("NA")
)

df_patient_flags = df_patient_flags.withColumn(
    "route_earliest",
    F.lit("NA")
)

df_patient_flags=df_patient_flags.withColumn("YearMonth",F.date_format("diagnosis_date_earliest", "yyyy-MM"))

df_patient_flags = df_patient_flags.withColumn(
    "imd_decile_group",
    F.when(F.col("IMD_Decile").isin("1", "2", "3"), "decile_1_to_3")
     .when(F.col("IMD_Decile").isin("4", "5", "6", "7"), "decile_4_to_7")
     .when(F.col("IMD_Decile").isin("8", "9", "10"), "decile_8_to_10")
     .otherwise("unknown")
)


df_patient_flags = df_patient_flags.withColumn(
    "age_10yr_band",
    F.when(F.col("Age") < 10, "0-9")
     .when((F.col("Age") >= 10) & (F.col("Age") < 20), "10-19")
     .when((F.col("Age") >= 20) & (F.col("Age") < 30), "20-29")
     .when((F.col("Age") >= 30) & (F.col("Age") < 40), "30-39")
     .when((F.col("Age") >= 40) & (F.col("Age") < 50), "40-49")
     .when((F.col("Age") >= 50) & (F.col("Age") < 60), "50-59")
     .when((F.col("Age") >= 60) & (F.col("Age") < 70), "60-69")
     .when((F.col("Age") >= 70) & (F.col("Age") < 80), "70-79")
     .when((F.col("Age") >= 80) & (F.col("Age") < 90), "80-89")
     .when(F.col("Age") >= 90, "90+")
     .otherwise("unknown")
)


# COMMAND ----------

for source_of_flag in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]:

    df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_from_{source_of_flag}_red_flag",
                                                F.date_diff(F.col("diagnosis_date_earliest"),
                                                            F.col(f"date_first_reported_{source_of_flag}_red_flag_in_last_{str(history_days)}_days")))

    df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_from_{source_of_flag}_amber_flag",
                                                F.date_diff(F.col("diagnosis_date_earliest"),
                                                            F.col(f"date_first_reported_{source_of_flag}_amber_flag_in_last_{str(history_days)}_days")))


# get list of column names for time to diagnosis from flag
list_time_diagnosis_red_flag_col_names = [f"time_diagnosis_from_{source_of_flag}_red_flag" for source_of_flag in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]]
list_time_diagnosis_amber_flag_col_names = [f"time_diagnosis_from_{source_of_flag}_amber_flag" for source_of_flag in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]]

df_patient_flags = df_patient_flags.withColumn("time_diagnosis_from_gp_referral_suspected",
                                               F.date_diff(F.col("diagnosis_date_earliest"), F.col(f"date_first_reported_gp_referral_suspected_in_last_{str(history_days)}_days")))


df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_from_first_reported_red_flag_in_last_{str(history_days)}_days",
                                               F.least(*[F.col(c) for c in list_time_diagnosis_red_flag_col_names]))

df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_from_first_reported_amber_flag_in_last_{str(history_days)}_days",
                                               F.least(*[F.col(c) for c in list_time_diagnosis_amber_flag_col_names]))


# COMMAND ----------

# get list of column names for count of flags in last history_days 

list_red_flag_count_columns = [f"number_of_times_{source_of_flag}_red_flag_in_last_" + str(history_days) + "_days" for source_of_flag in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]]
list_amber_flag_count_columns = [f"number_of_times_{source_of_flag}_amber_flag_in_last_" + str(history_days) + "_days" for source_of_flag in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]]

df_patient_flags = df_patient_flags.fillna(0, subset = list_red_flag_count_columns + list_amber_flag_count_columns)

df_patient_flags = df_patient_flags.withColumn("total_number_red_flag_symptoms_in_last_" + str(history_days) + "_days",
                                               reduce(add, [F.col(x) for x in list_red_flag_count_columns]))

df_patient_flags = df_patient_flags.withColumn("total_number_amber_flag_symptoms_in_last_" + str(history_days) + "_days",
                                               reduce(add, [F.col(x) for x in list_amber_flag_count_columns]))

list_red_flag_count_columns.append("total_number_red_flag_symptoms_in_last_" + str(history_days) + "_days")
list_amber_flag_count_columns.append("total_number_amber_flag_symptoms_in_last_" + str(history_days) + "_days")

# Binary flags

df_patient_flags = df_patient_flags.withColumn(f"flag_any_red_flag_in_last_{str(history_days)}_days",
                                               F.when((F.col(f"total_number_red_flag_symptoms_in_last_{str(history_days)}_days")>0), 1).otherwise(0))

df_patient_flags = df_patient_flags.withColumn(f"flag_any_amber_flag_in_last_{str(history_days)}_days",
                                               F.when((F.col(f"total_number_amber_flag_symptoms_in_last_{str(history_days)}_days")>0), 1).otherwise(0))

df_patient_flags = df_patient_flags.withColumn(f"flag_gp_referral_suspected_in_last_{str(history_days)}_days",
                                               F.when((F.col(f"date_first_reported_gp_referral_suspected_in_last_{str(history_days)}_days").isNotNull()), 1).otherwise(0))

df_patient_flags = df_patient_flags.withColumn(f"flag_chest_xray_in_last_{str(history_days)}_days",
                                               F.when((F.col(f"date_last_reported_chest_xray_in_last_" + str(history_days) + "_days").isNotNull()), 1).otherwise(0))       

df_patient_flags = df_patient_flags.fillna(0, subset = [f"number_of_times_chest_xray_in_last_{str(history_days)}_days"])                             

# COMMAND ----------

# MAGIC %md
# MAGIC ## conditions for referral

# COMMAND ----------

# identify earliest reporting of each symptom and count 

all_symptoms = config_pathways.cancer_site_mappings[cancer_site]["unexplained_symptoms_NICE_guidelines"] + config_pathways.cancer_site_mappings[cancer_site]["critical_symptoms_NICE_guidelines"] + config_pathways.cancer_site_mappings[cancer_site]["medication_flags"]
sources_of_flags = config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]
all_columns = df_patient_flags.columns

for symptom in all_symptoms:

    print("processing: ", symptom)
    all_cols_related_to_symptom = [col for col in all_columns if "_" + symptom + "_" in col.lower()]
    date_related_symptom_col = [col for col in all_cols_related_to_symptom if "date_first_reported" in col.lower()]
    count_related_symptom_col = [col for col in all_cols_related_to_symptom if "number_of_times" in col.lower()]

    print(date_related_symptom_col)
    print(count_related_symptom_col)

    df_patient_flags = df_patient_flags.fillna(0, subset = count_related_symptom_col)

    for col in count_related_symptom_col:
        for source in sources_of_flags:
            if source in col:
                # create a flag to identify if this was symptom was reported in this setting
                df_patient_flags = df_patient_flags.withColumn(f"flag_{symptom}_reported_in_{source}_in_last_{str(history_days)}_days",
                                                            F.when(F.col(col) >= 1, 1).otherwise(0))
       

    if len(date_related_symptom_col) == 0:
        print("no date columns found for symptom: ", symptom)
        continue

    elif len(date_related_symptom_col) == 1:
        df_patient_flags = df_patient_flags.withColumn(f"date_first_reported_{symptom}_in_last_{str(history_days)}_days_overall",
                                                   F.col(date_related_symptom_col[0]))

        df_patient_flags = df_patient_flags.withColumn(f"number_of_times_{symptom}_reported_in_last_{str(history_days)}_days_overall",
                                                    F.col(count_related_symptom_col[0]))
        
        # create a flag to identify if this was symptom was reported at all in any setting
        df_patient_flags = df_patient_flags.withColumn(f"flag_{symptom}_reported_in_last_{str(history_days)}_days",
                                                       F.when(F.col(f"number_of_times_{symptom}_reported_in_last_{str(history_days)}_days_overall") >= 1, 1).otherwise(0))

    else:
        # get the earliest date per feature across all symptoms
        df_patient_flags = df_patient_flags.withColumn(f"date_first_reported_{symptom}_in_last_{str(history_days)}_days_overall",
                                                    F.least(*[F.col(c) for c in date_related_symptom_col]))

        # sum all the count columns
        df_patient_flags = df_patient_flags.withColumn(f"number_of_times_{symptom}_reported_in_last_{str(history_days)}_days_overall",
                                                    reduce(add, [F.col(x) for x in count_related_symptom_col]))

        # create a flag to identify if this was symptom was reported at all in any setting
        df_patient_flags = df_patient_flags.withColumn(f"flag_{symptom}_reported_in_last_{str(history_days)}_days",
                                                       F.when(F.col(f"number_of_times_{symptom}_reported_in_last_{str(history_days)}_days_overall") >= 1, 1).otherwise(0))



# COMMAND ----------

# MAGIC %md
# MAGIC - Get the count and earliest date of any of the unexplained symptoms
# MAGIC - Get the date of when two unexplained symptoms have been reported

# COMMAND ----------

# count how many of the unexplained_symptoms_NICE_guidelines symptoms are present for the patient

column_names_unexplained_symptoms_NICE_guidelines_overall_counts = []
column_names_unexplained_symptoms_NICE_guidelines_earliest_date = []


for col in df_patient_flags.columns:
    for symptom in config_pathways.cancer_site_mappings[cancer_site]["unexplained_symptoms_NICE_guidelines"]:
        if symptom + "_" in col.lower() and "overall" in col.lower():
            if "number_of_times_" in col.lower():
                column_names_unexplained_symptoms_NICE_guidelines_overall_counts.append(col)
            if "date_first_reported_" in col.lower():
                column_names_unexplained_symptoms_NICE_guidelines_earliest_date.append(col)

print("columns for count: ", column_names_unexplained_symptoms_NICE_guidelines_overall_counts)
print("\n")
print("columns for date", column_names_unexplained_symptoms_NICE_guidelines_earliest_date)

# count how many of the unexplained_symptoms_NICE_guidelines symptoms are present for the patient
nonzero_count = sum(
    F.when(F.coalesce(F.col(c), F.lit(0)) != 0, F.lit(1)).otherwise(F.lit(0))
    for c in column_names_unexplained_symptoms_NICE_guidelines_overall_counts
)

df_patient_flags = df_patient_flags.withColumn(f"number_of_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days", nonzero_count)

# get the earliest date for the unexplained symptom

# build an array of dates 
array_dates_unexplained_symptoms = F.array(*[F.col(c) for c in column_names_unexplained_symptoms_NICE_guidelines_earliest_date])

df_patient_flags = df_patient_flags.withColumn(f"array_dates_unexplained_symptoms",array_dates_unexplained_symptoms)

valid = F.expr("array_sort(filter(array_dates_unexplained_symptoms, x -> x is not null))")
smallest_date = F.when(F.size(valid) >= 1, F.element_at(valid, 1))
second_smallest_date = F.when(F.size(valid) >= 2, F.element_at(valid, 2))

df_patient_flags = df_patient_flags.withColumn(f"earliest_date_one_reported_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days",smallest_date)
df_patient_flags = df_patient_flags.withColumn(f"earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days",second_smallest_date)


# COMMAND ----------

# MAGIC %md
# MAGIC - Get the count and earliest date of any critical symptom
# MAGIC

# COMMAND ----------

# identify the earliest date for each

column_names_critical_symptoms_NICE_guidelines_overall_counts = []
column_names_critical_symptoms_NICE_guidelines_earliest_date = []

for col in df_patient_flags.columns:
    for symptom in config_pathways.cancer_site_mappings[cancer_site]["critical_symptoms_NICE_guidelines"]:
        if symptom + "_" in col.lower() and "overall" in col.lower():
            if "number_of_times_" in col.lower():
                column_names_critical_symptoms_NICE_guidelines_overall_counts.append(col)
            if "date_first_reported_" in col.lower():
                column_names_critical_symptoms_NICE_guidelines_earliest_date.append(col)


print("columns for count: ", column_names_critical_symptoms_NICE_guidelines_overall_counts)
print("\n")
print("columns for date", column_names_critical_symptoms_NICE_guidelines_earliest_date)

# count how many of the critical_symptoms_NICE_guidelines symptoms are present for the patient
nonzero_count = sum(
    F.when(F.coalesce(F.col(c), F.lit(0)) != 0, F.lit(1)).otherwise(F.lit(0))
    for c in column_names_critical_symptoms_NICE_guidelines_overall_counts
)

df_patient_flags = df_patient_flags.withColumn(f"number_of_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days", nonzero_count)


# get the earliest date for the critical_symptoms_NICE_guidelines symptoms
df_patient_flags = df_patient_flags.withColumn(f"date_first_reported_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days",
                                               F.least(*[F.col(c) for c in column_names_critical_symptoms_NICE_guidelines_earliest_date]))






# COMMAND ----------

# MAGIC %md
# MAGIC #### Smoker and 1 unexplained symptom

# COMMAND ----------

# flag if patient has smoked previously, and any one of unexplained_symptoms_NICE_guidelines is true
df_patient_flags = df_patient_flags.withColumn(f"flag_smoker_with_unexplained_symptoms_in_last_{str(history_days)}_days",
                                               F.when((F.col("Smoking_Flag") == 1) & (F.col(f"number_of_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days") > 0), 1).otherwise(0))

# find earliest date when the condition is met
df_patient_flags = df_patient_flags.withColumn(f"earliest_date_smoker_with_unexplained_symptoms_in_last_{str(history_days)}_days",
                                               F.when(F.col(f"flag_smoker_with_unexplained_symptoms_in_last_{str(history_days)}_days")==1,
                                                      F.col(f"earliest_date_one_reported_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days" ) 
                                                      ).otherwise(F.lit(None))
                                               )

# FIND SOURCE OF EARLIEST REPORT

# COMMAND ----------

# MAGIC %md
# MAGIC #### One or more unexplained symptom

# COMMAND ----------

# flag if patient has one unexplained_symptoms_NICE_guideline
df_patient_flags = df_patient_flags.withColumn(f"flag_one_or_more_unexplained_symptoms_in_last_{str(history_days)}_days",
                                               F.when(F.col(f"number_of_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days") >= 1, 1).otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Two or more unexplained symptoms

# COMMAND ----------

# flag if patient has two unexplained_symptoms_NICE_guideline
df_patient_flags = df_patient_flags.withColumn(f"flag_two_or_more_unexplained_symptoms_in_last_{str(history_days)}_days",
                                               F.when(F.col(f"number_of_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days") >= 2, 1).otherwise(0))


# date column name is f"earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days

# COMMAND ----------

# MAGIC %md
# MAGIC ### At least one critical symptom

# COMMAND ----------

# flag if patient has has at least one critical_symptoms_NICE_guidelines
df_patient_flags = df_patient_flags.withColumn(f"flag_atleast_one_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days",
                                               F.when(F.col(f"number_of_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days") > 0, 1).otherwise(0))

# the date column for this would be f"date_first_reported_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get time to diagnosis

# COMMAND ----------

df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_for_smoker_with_unexplained_symptoms",
                                               F.date_diff(F.col("diagnosis_date_earliest"),
                                                           F.col(f"earliest_date_smoker_with_unexplained_symptoms_in_last_{str(history_days)}_days")))

df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_for_two_or_more_unexplained_symptoms",
                                               F.date_diff(F.col("diagnosis_date_earliest"),
                                                           F.col(f"earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days")))

df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_for_one_critical_symptom",
                                               F.date_diff(F.col("diagnosis_date_earliest"),
                                                           F.col(f"date_first_reported_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days")))


# COMMAND ----------

df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_from_earliest_xray",
                                               F.date_diff(F.col("diagnosis_date_earliest"),
                                                           F.col("date_first_reported_chest_xray_in_last_" + str(history_days) + "_days")))

df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_from_latest_xray",
                                               F.date_diff(F.col("diagnosis_date_earliest"),
                                                           F.col(f"date_last_reported_chest_xray_in_last_" + str(history_days) + "_days")))


# COMMAND ----------

for symptom in all_symptoms:
    for col in df_patient_flags.columns:
        if symptom in col and "date_first_reported" in col:
            df_patient_flags = df_patient_flags.withColumn(f"time_diagnosis_from_" + col,
                                                           F.date_diff(F.col("diagnosis_date_earliest"),
                                                           F.col(col))
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ### time to chest x-ray

# COMMAND ----------

# drop date from df_patient_flags
df_patient_flags = df_patient_flags.drop("date")

df_xray_patients = df_patient_flags.join(df_chest_xray_combined, on = "Patient_ID", how="inner")

# for each key milestone, identify days between the condition being met and the chest x-ray

milestones = [f"earliest_date_one_reported_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days",
              f"earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days",
              f"earliest_date_smoker_with_unexplained_symptoms_in_last_{str(history_days)}_days",
              f"date_first_reported_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days",
              ]

for milestone_date in milestones:
    df_xray_patients_flag = df_xray_patients.withColumn("days_between_milestone_and_xray",
                                                        F.date_diff(F.col("date"),
                                                                    F.col(milestone_date)))

    df_xray_patients_flag = df_xray_patients_flag.filter(F.col("days_between_milestone_and_xray")>=-10)

    df_xray_patients_min_days = df_xray_patients_flag.groupBy("Patient_ID").agg(
        F.min(F.col("days_between_milestone_and_xray")).alias("days_xray_to_" + milestone_date)
    )

    df_patient_flags = df_patient_flags.join(df_xray_patients_min_days, on = "Patient_ID", how = "left")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Export
# MAGIC

# COMMAND ----------

display(df_patient_flags)

# COMMAND ----------

df_patient_flags.columns

# COMMAND ----------

df_patient_flags.cache()

# COMMAND ----------

fullPath_output=""

df_patient_flags.write.mode("overwrite").parquet(
    fullPath_output,
)


# COMMAND ----------

fullPath_output=""

df_gp_red_flags.write.mode("overwrite").parquet(
    fullPath_output,
)

# COMMAND ----------

fullPath_output=""

df_gp_red_flags_ref.write.mode("overwrite").parquet(
    fullPath_output,
)

# COMMAND ----------

fullPath_output=""

df_gp_amber_flags.write.mode("overwrite").parquet(
    fullPath_output,
)

# COMMAND ----------

fullPath_output=""

df_acute_red_flag.write.mode("overwrite").parquet(
    fullPath_output,
)

# COMMAND ----------

fullPath_output=""

df_referral.write.mode("overwrite").parquet(
    fullPath_output,
)

# COMMAND ----------

fullPath_output=""

df_screening_snomed.write.mode("overwrite").parquet(
    fullPath_output,
)
