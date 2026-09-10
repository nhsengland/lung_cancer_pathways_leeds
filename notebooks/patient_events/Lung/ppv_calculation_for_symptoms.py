# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType
import src.cancer_late.config  as config
import src.cancer_late.config_pathways  as config_pathways
from src.cancer_late import processing
from src.cancer_late.utils import read_parquet_file, read_csv_file
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType
from datetime import date
import pandas as pd
import numpy as np


# COMMAND ----------

cancer_site = "Lung"
recreate_activity_table = False
write_output = True

version = config_pathways.cancer_site_mappings[cancer_site]["run_version"]

earliest_cancer_date = "2022-05-01"
latest_cancer_date = "2025-01-01"

max_days_to_diagnosis_from_symptom = 365

def change_one_year_same_md(d: date, num_years) -> date:
    try:
        return d.replace(year=d.year + num_years)
    except ValueError:
        # This happens for Feb 29 on a non-leap year.
        # A common choice is to fallback to Feb 28.
        return d.replace(month=2, day=28, year=d.year + num_years)

d = date.fromisoformat(latest_cancer_date)
d_next = change_one_year_same_md(d, num_years = -1)
latest_activity_date = d_next.isoformat()
print("latest_activity_date", latest_activity_date) 

days_ppv = 365


list_of_symptoms_non_critical = config_pathways.cancer_site_mappings["Lung"]["unexplained_symptoms_NICE_guidelines"] 

list_of_symptoms_critical = config_pathways.cancer_site_mappings["Lung"]["critical_symptoms_NICE_guidelines"] 

list_of_symptoms_incl_prescriptions = list_of_symptoms_non_critical + list_of_symptoms_critical + config_pathways.cancer_site_mappings["Lung"]["medication_flags"]


# COMMAND ----------

# MAGIC %md
# MAGIC # Mapping files

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

df_gp_amber_flags_ref = df_gp_amber_flags_ref.withColumn("grouping", F.lit("amber_flag"))
df_gp_red_flags_ref = df_gp_red_flags_ref.withColumn("grouping", F.lit("red_flag"))

# COMMAND ----------

if cancer_site == "Lung":
    df_mapping_chest_xray = read_csv_file(containerName =config.containerName_platinum, 
                                        lakeName=config.lakeName,
                                        filePath= config_pathways.cancer_site_mappings[cancer_site]["chest_xray_snomed"])
    
    medication_table_paths = config_pathways.cancer_site_mappings[cancer_site]["medication_code_tables"]
    
    i = 0 

    for medication_name, medication_table_path in medication_table_paths.items():
        df = read_csv_file(containerName =config.containerName_platinum, 
                                            lakeName=config.lakeName,
                                            filePath= medication_table_path)
        

        df = df.withColumn("grouping", F.lit(medication_name))
        df = df.withColumn("symptom_name", F.lit(medication_name))

        if i==0:
            df_medication_mapping = df

        else:
            df_medication_mapping = df_medication_mapping.unionByName(df)     

        i = i+1

# COMMAND ----------


df_111_mapping = read_csv_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= config_pathways.cancer_site_mappings[cancer_site]["flags_111"])

df_ecds_mapping = read_csv_file(containerName =config.containerName_platinum, 
                                lakeName=config.lakeName,
                                filePath= config_pathways.cancer_site_mappings[cancer_site]["flags_ecds"])

df_ecds_mapping_red_flag = df_ecds_mapping.filter(F.col("grouping")=="red_flag").withColumnRenamed("SNOMED_UK_Preferred_Term", "ecds_chief_complaint")

df_ecds_mapping_amber_flag = df_ecds_mapping.filter(F.col("grouping")=="amber_flag").withColumnRenamed("SNOMED_UK_Preferred_Term", "ecds_chief_complaint")

# COMMAND ----------

df_icd10_codes_red_flag = read_csv_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= config_pathways.cancer_site_mappings[cancer_site]["icd10_red_flag_codes"])

df_icd10_codes_amber_flag = read_csv_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= config_pathways.cancer_site_mappings[cancer_site]["icd10_amber_flag_codes"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Cancer registry

# COMMAND ----------

# load cancer registry and process
# data granularity is ID-cancer group. Per patient, per cancer group, identify key metrics such as dates of diagnosis and stage
df_cancer_by_group = processing.process_cancer_datasets(config.containerName_bronze,
                                                        config.lakeName,
                                                        config.filePath_cancer_registration_registry,
                                                        config.filePath_cancer_registration_rapid,
                                                        config.filePath_deaths)

# if route_earliest is TWW, put "USC"
# if route_earliest is "NOT AVAILABLE FOR DIAGNOSIS YEAR" or "ROUTE NOT CLASSIFIED" put "Unknown"
# if route_earliest is "cause_of_death" or "DCO" put "death_certificate_only"

df_cancer_by_group = (
    df_cancer_by_group
    .withColumn(
        "route_earliest",
        F.when(F.col("route_earliest") == "TWW", "USC")
         .when(F.col("route_earliest").isin("cause_of_death", "DCO"), "death_certificate_only")
         .when(F.col("route_earliest").isin("NOT AVAILABLE FOR DIAGNOSIS YEAR", "ROUTE NOT CLASSIFIED"), "Unknown")
         .otherwise(F.col("route_earliest"))
    )
)

# get the date of the earliest recorded cancer diagnosis, per patient
df_id_first_cancer = df_cancer_by_group.groupBy("PSEUDO_NHS_NUMBER").agg(F.min("diagnosis_date_earliest").alias("diagnosis_date_earliest"))

# keep cancers only from cancer site
df_cancer_site = df_cancer_by_group.filter(F.col("Cancer_Group")==cancer_site)

# only keep cancers with a diagnosis date after april 2022 and before the end of the registry data 
df_cancer_site = df_cancer_site.filter((F.col("diagnosis_date_earliest") >= earliest_cancer_date) & 
                                       (F.col("diagnosis_date_earliest") < latest_cancer_date))

# keep cancers where it is the first cancer
df_cancer_site = df_cancer_site.join(df_id_first_cancer,
                                     on = ["PSEUDO_NHS_NUMBER", "diagnosis_date_earliest"],
                                     how = "inner")

total_num_pathways = df_cancer_site.count()
total_num_unique_patients = df_cancer_site.select("PSEUDO_NHS_NUMBER").distinct().count()

print("Total number of patients in the dataset: ", total_num_unique_patients)
print("Total number of records in the dataset: ", total_num_pathways)

# get the first character of the tumour stage (1,2,3 or 4). This removes the information such as stage 2A etc
df_cancer_site = df_cancer_site.withColumn("tumour_stage_earliest_first_char",
                                           F.substring("tumour_stage_earliest", 1, 1))

df_cancer_site = df_cancer_site.withColumnRenamed("PSEUDO_NHS_NUMBER", "Patient_ID")

# COMMAND ----------

# MAGIC %md
# MAGIC # Cohort tables

# COMMAND ----------

# MAGIC %md
# MAGIC Identify the earliest cohort table prior to the patient's cancer diagnosis
# MAGIC - df_all_cohorts --> one row per patient. Identifies all the cohort tables a patient is in through columns
# MAGIC - df_multi_cohorts --> one row per patient per cohort
# MAGIC

# COMMAND ----------

df_multi_cohorts = read_parquet_file(containerName =config.containerName_platinum, 
                                lakeName=config.lakeName,
                                filePath= f"")

df_all_cohorts = read_parquet_file(containerName =config.containerName_platinum, 
                                lakeName=config.lakeName,
                                filePath= f"")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create new smoking flag

# COMMAND ----------

df_pcp_emis = read_parquet_file(containerName =config.containerName_bronze, 
                             lakeName= config.lakeName,
                             filePath= config.filePath_emis) 

df_pcp_s1 = read_parquet_file(containerName =config.containerName_bronze, 
                           lakeName= config.lakeName,
                           filePath= config.filePath_s1)

df_gp_events = processing.process_gp_data_emis_s1(df_pcp_emis,df_pcp_s1 )

smoking_cessation_df = read_csv_file(containerName =config.containerName_platinum, 
                           lakeName= config.lakeName,
                           filePath= config_pathways.smoking_cessation_snomed)

# COMMAND ----------

join_condition = (df_gp_events.SnomedCode == smoking_cessation_df.code)

smoking_events = df_gp_events.join(smoking_cessation_df, on = join_condition, how="inner")

df_smoking_patients = smoking_events.select("Patient_ID").distinct()
df_smoking_patients = df_smoking_patients.withColumn("Smoking_Flag", F.lit(True))

df_all_patients = df_gp_events.select("Patient_ID").distinct()

df_smoking_flags = df_all_patients.join(df_smoking_patients, on="Patient_ID", how="left")
df_smoking_flags = df_smoking_flags.fillna(False, subset="Smoking_Flag")
df_smoking_flags = df_smoking_flags.dropDuplicates()

# COMMAND ----------

if "Smoking_Flag" in df_multi_cohorts.columns:
    df_multi_cohorts = df_multi_cohorts.drop("Smoking_Flag")
    
df_multi_cohorts = df_multi_cohorts.join(df_smoking_flags, on="Patient_ID", how="left")
df_multi_cohorts = df_multi_cohorts.fillna(False, subset="Smoking_Flag")

# COMMAND ----------

df_all_cohorts.count()

# COMMAND ----------

list_cohorts_needed = []
list_cohort_ages = []

for col in df_all_cohorts.columns:
    if "Cohort_Date" in col:
        date_of_cohort = col.split("Date_")[1]

        if date_of_cohort>=earliest_cancer_date and date_of_cohort<= latest_cancer_date:
            list_cohorts_needed.append(col)

    elif "Age" in col:
        date_of_cohort = col.split("Age_")[1]
        
        if date_of_cohort>=earliest_cancer_date and date_of_cohort<= latest_cancer_date:
            list_cohort_ages.append(col)

list_cohorts_needed

# COMMAND ----------

df_patients_with_complete_data = df_all_cohorts.dropna(subset=list_cohorts_needed)
df_patients_with_complete_data = df_patients_with_complete_data.filter(F.col("Age_2023-03-31")>=40)
df_patients_with_complete_data.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Events table

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cancer diagnosis event

# COMMAND ----------

df_cancer_site = df_cancer_site.withColumn("dataset", F.lit("Cancer_registry"))
df_cancer_site = df_cancer_site.withColumn("date", F.col("diagnosis_date_earliest"))
df_cancer_site = df_cancer_site.withColumn("description", 
                           F.concat_ws("", F.lit("Tumour site: "), F.col("tumour_site_earliest_diagnosis"),
                                    F.lit(" | ") , F.lit("Stage: "), F.col("tumour_stage_earliest"),
                                    F.lit(" | ") , F.lit("Route: "), F.col("route_earliest"),
                                    F.lit(" | ") , F.lit("Age at diagnosis: "), F.col("age_earliest_diagnosis")))



# COMMAND ----------

# ensure cancer patients are in the cohort tables in the 1 year before their diagnosis

# add the diagnosis date information to the multi cohort table
df_multi_cohorts_selected_ids = df_multi_cohorts.join(df_cancer_site.select(["Patient_ID", "diagnosis_date_earliest"]),
                                                      on = "Patient_ID",
                                                      how="inner")

df_multi_cohorts_selected_ids = df_multi_cohorts_selected_ids.withColumn(
    "days_between_diagnosis_and_cohort",
    F.datediff(
        F.col("diagnosis_date_earliest"),
        F.col("Cohort_Date")
    )
)

df_multi_cohorts_selected_ids = df_multi_cohorts_selected_ids.withColumn(
    "history_start",
    F.date_sub(F.col("diagnosis_date_earliest"), 365)
)

# only keep cohort data in the year prior to the diagnosis
df_multi_cohorts_selected_ids_in_last_year = df_multi_cohorts_selected_ids.filter(
    (F.col("days_between_diagnosis_and_cohort")>=0) & (F.col("days_between_diagnosis_and_cohort")<365))

# count how many cohorts the patient was included in the last 1 year
df_count_cohorts_in_last_year = df_multi_cohorts_selected_ids_in_last_year.groupby(["Patient_ID"]).count()

df_four_quarters_in_last_year = df_count_cohorts_in_last_year.filter(F.col("count")>=4)

# updated cancer site table -> only those with at least 4 quarters in last year

df_cancer_site_complete_data = df_cancer_site.join(df_four_quarters_in_last_year.select("Patient_ID"), on = "Patient_ID", how="inner")

df_cancer_site_complete_data.count()

# COMMAND ----------

# identify and keep the latest cohort 
window_spec = Window.partitionBy("Patient_ID").orderBy(F.col("days_between_diagnosis_and_cohort").asc())
df_latest_cohort = df_multi_cohorts_selected_ids_in_last_year.withColumn("row_num", F.row_number().over(window_spec)).filter(F.col("row_num") == 1).drop("row_num")

# remove any patients with a LTC_Cancer flag
df_latest_cohort = df_latest_cohort.filter(F.col("LTC_Cancer")==0)

df_cancer_site_complete_data_no_ltc_flag = df_cancer_site_complete_data.join(df_latest_cohort.select("Patient_ID", "Age", "Smoking_Flag"), on = "Patient_ID", how="inner")
df_cancer_site_complete_data_no_ltc_flag = df_cancer_site_complete_data_no_ltc_flag.filter(F.col("Age")>=40)
print(df_cancer_site_complete_data_no_ltc_flag.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create All activity table

# COMMAND ----------

if recreate_activity_table == True:

    df_sct_concept_definitions = read_parquet_file(containerName =config.containerName_bronze, 
                                                lakeName= config.lakeName,
                                                filePath= "") 

    df_chosen_snomed_description = (
        df_sct_concept_definitions
        # Keep only rows where the description type is 'Preferred Term'
        .filter(F.col("Description_Type") == "Preferred Term")
        # Keep only rows where the description is active
        .filter(F.col("Active_Description") == True)
        # Assign priority: 1 if both concept and description are active, else 2
        .withColumn("priority", F.when((F.col("Active_Concept") == True) & (F.col("Active_Description") == True), 1).otherwise(2))
        # Assign row numbers within each Concept_ID partition, ordered by priority
        .withColumn("row_num", F.row_number().over(Window.partitionBy("Concept_ID").orderBy("priority")))
        # Keep only the top-priority row for each Concept_ID
        .filter(F.col("row_num") == 1)
        # Drop helper columns
        .drop("priority", "row_num")
    )

    # only keeping GP events for cancer patients
    df_gp_events = df_gp_events.withColumn("dataset", F.lit("gp_events"))
    df_gp_events = df_gp_events.withColumn("date", F.col("Attendance_Date"))

    df_gp_events = df_gp_events.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                    df_gp_events.SnomedCode == df_chosen_snomed_description.Concept_ID,
                                    how="left" )

    df_gp_events = df_gp_events.filter(F.col("SnomedCode") != "-1") # remove where code is -1

    # remove duplicates
    df_gp_events = df_gp_events.dropDuplicates(["Patient_ID", "date", "SnomedCode"])

    df_gp_events = df_gp_events.withColumn("description", F.col("Term"))

    # SUS
    df_sus_all = read_parquet_file(containerName =config.containerName_bronze, 
                                lakeName=config.lakeName,
                                filePath= config.filePath_sus)

    df_sus_all = df_sus_all.withColumn("dataset", F.lit("SUS_Activity_Extract"))
    df_sus_all = df_sus_all.withColumn("date", F.col("Attendance_Date"))

    all_columns = df_sus_all.columns

    # Define diagnosis/procedure column pairs
    diagnosis_cols = ['Primary_Diagnosis'] + [(f"Secondary_Diagnosis_{i}") for i in range(1, 13)]
    procedure_cols = ['Primary_Procedure_Code'] + [(f"Secondary_Procedure_Code_{i}") for i in range(1, 13)]
    hrg_cols = ['HRG_Code']

    # columns to keep after wide to long transformation (all columns minus the ones going to rows)
    demographic_cols = list(set(all_columns)- set(diagnosis_cols)- set(procedure_cols)- set(hrg_cols))

    # wide to long transformation

    # Build stack() expression dynamically
    expr_parts = []
    for event_type, cols in [("Diagnosis_SUS", diagnosis_cols), ("Procedure_SUS", procedure_cols), ('HRG_SUS', hrg_cols)]:
        for code_col in cols:
            expr_parts.append(f"'{event_type}', {code_col}")

    stack_expr = f"stack({len(expr_parts)}, {', '.join(expr_parts)}) as (Event_Type, Event_Code)"

    # Perform unpivot
    df_sus_all_events = (
        df_sus_all.selectExpr(*demographic_cols, stack_expr)
        .filter("Event_Code is not null")
    )

    df_sus_all_events = df_sus_all_events.withColumn("Event_Code", F.regexp_replace(F.col("Event_Code"), r"\.", ""))

    # remove duplicates 
    df_sus_all_events = df_sus_all_events.dropDuplicates(["Patient_ID", "date", "Event_Code", "Event_Type"])

    df_icd10_ref = read_parquet_file(containerName =config.containerName_bronze, 
                                lakeName= config.lakeName,
                                filePath= config.filePath_icd10) 

    df_icd10_ref_cat_3 = processing.process_icd10_ref(df_icd10_ref)

    df_sus_all_events = df_sus_all_events.withColumn(
                                        "Diagnosis_type",
                                    F.when(
                                        F.col("Event_Type") == "Diagnosis_SUS",
                                        F.when(F.regexp_extract(F.col("Event_Code"), r'^[A-Z][0-9][0-9A-Z]?$|^[A-Z][0-9][0-9A-Z]+$', 0) != "", "ICD10")
                                    .when(F.regexp_extract(F.col("Event_Code"), r'^[0-9]{6,18}$', 0) != "", "SNOMED")
                                    .otherwise("Other")
                                    ).otherwise("")
    )

    df_sus_all_events = df_sus_all_events.withColumn(
        "Diagnosis_SUS_icd10_3_char",
        F.when(
            (F.col("Diagnosis_type") == "ICD10"),
            F.substring(F.col("Event_Code"), 1, 3)
        ).otherwise("")
        )

    df_sus_all_events = df_sus_all_events.join(df_icd10_ref_cat_3.select(["Alt_Code_3_char", "Description"]),
                                            df_sus_all_events.Diagnosis_SUS_icd10_3_char == df_icd10_ref_cat_3.Alt_Code_3_char,
                                            "left")

    df_sus_all_events = df_sus_all_events.withColumnRenamed("Description", "ICD10_description")

    df_opcs_ref = read_parquet_file(containerName =config.containerName_bronze, 
                                lakeName= config.lakeName,
                                filePath= "") 

    w_latest = Window.partitionBy("Code_Without_Decimal").orderBy(F.desc("Effective_to"))

    df_opcs_ref_latest = (
        df_opcs_ref
        .withColumn("row_num", F.row_number().over(w_latest))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
    )

    df_sus_all_events = df_sus_all_events.withColumn(
                                        "Procedure_type",
                                    F.when(
                                        F.col("Event_Type") == "Procedure_SUS",
                                        F.when(F.regexp_extract(F.col("Event_Code"), r'^[A-Z][0-9][0-9A-Z]?$|^[A-Z][0-9][0-9A-Z]+$', 0) != "", "OPCS")
                                    .when(F.regexp_extract(F.col("Event_Code"), r'^[0-9]{6,18}$', 0) != "", "SNOMED")
                                    .otherwise("Other")
                                    ).otherwise("")
    )

    df_sus_all_events = df_sus_all_events.join(df_opcs_ref_latest.select(["Code_Without_Decimal", "Title"]),
                                            (df_sus_all_events.Event_Code == df_opcs_ref_latest.Code_Without_Decimal) & (df_sus_all_events.Procedure_type == "OPCS"),
                                            "left")

    df_sus_all_events = df_sus_all_events.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                            (df_sus_all_events.Event_Code == df_chosen_snomed_description.Concept_ID) & ((df_sus_all_events.Diagnosis_type == "SNOMED") | ((df_sus_all_events.Procedure_type == "SNOMED"))),
                                            "left")

    df_hrg = read_parquet_file(containerName =config.containerName_bronze, 
                                lakeName= config.lakeName,
                                filePath= "") 

    df_hrg_latest = df_hrg.filter(F.col("Is_Latest")==1)

    df_sus_all_events = df_sus_all_events.join(
        df_hrg_latest.select(["HRG_Code", "HRG_Name"]),
        (df_sus_all_events.Event_Code == df_hrg_latest.HRG_Code) &
        (df_sus_all_events.Event_Type=="HRG_SUS"),
        "left"
    )

    df_sus_all_events = df_sus_all_events.withColumn("description", 
                                    F.when(
                                        (F.col("Event_Type") == "Diagnosis_SUS") & (F.col("Diagnosis_type") == "ICD10") & (F.col("ICD10_Description").isNotNull()),
                                        F.concat_ws("",F.lit("Diagnosis: "), F.col("ICD10_Description"), F.lit(" | ") ,F.lit("Type: "), F.col("Record_Classification"))
                                    ).when(
                                        (F.col("Event_Type") == "Procedure_SUS") & (F.col("Procedure_type") == "OPCS") & (F.col("Title").isNotNull()),
                                        F.concat_ws("",F.lit("Procedure: "), F.col("Title"), F.lit(" | ") ,  F.lit("Type: "), F.col("Record_Classification"))
                                    ).when(
                                        (F.col("Event_Type") == "Diagnosis_SUS") & (F.col("Diagnosis_type") == "SNOMED") & (F.col("Term").isNotNull()),
                                        F.concat_ws("",F.lit("Diagnosis: "), F.col("Term"), F.lit(" | ") ,  F.lit("Type: "), F.col("Record_Classification"))
                                    ).when(
                                        (F.col("Event_Type") == "Procedure_SUS") & (F.col("Procedure_type") == "SNOMED") & (F.col("Term").isNotNull()),
                                        F.concat_ws("",F.lit("Procedure: "), F.col("Term"), F.lit(" | ") ,  F.lit("Type: "), F.col("Record_Classification"))
                                    ).when(
                                        (F.col("Event_Type") == "HRG_SUS") & (F.col("HRG_Name").isNotNull()),
                                        F.concat_ws("",F.lit("HRG: "), F.col("HRG_Name"), F.lit(" | ") , F.lit("Type: "), F.col("Record_Classification"))
                                    )                           
                                    .otherwise(F.lit(""))  # or F.lit('') if you prefer empty string
    )

    df_acute = read_parquet_file(containerName =config.containerName_bronze, 
                                lakeName=config.lakeName,
                                filePath=config.filePath_acute)

    df_acute = df_acute.withColumn("dataset", F.lit("Acute_All"))
    df_acute = df_acute.withColumn("date", F.col("Attendance_Date"))

    all_columns = df_acute.columns

    # Define diagnosis/procedure column pairs
    hrg_cols = ["Dimention_5"]
    diagnosis_cols = ["Dimention_6"]
    procedure_cols = ["Dimention_7"]


    # columns to keep after wide to long transformation (all columns minus the ones going to rows)
    demographic_cols = list(set(all_columns)- set(hrg_cols) - set(diagnosis_cols) - set(procedure_cols))

    # wide to long transformation

    # Build stack() expression dynamically
    expr_parts = []
    for event_type, cols in [("Diagnosis_ACUTE", diagnosis_cols), ("Procedure_ACUTE", procedure_cols), ('HRG_ACUTE', hrg_cols)]:
        for code_col in cols:
            expr_parts.append(f"'{event_type}', {code_col}")

    stack_expr = f"stack({len(expr_parts)}, {', '.join(expr_parts)}) as (Event_Type, Event_Code)"

    # Perform unpivot
    df_acute_all_events = (
        df_acute.selectExpr(*demographic_cols, stack_expr)
        .filter("Event_Code is not null")
    )

    df_acute_all_events = df_acute_all_events.withColumn("Event_Code", F.regexp_replace(F.col("Event_Code"), r"\.", ""))

    df_acute_all_events = df_acute_all_events.dropDuplicates(["Patient_ID", "date", "Event_Code", "Event_Type"])

    df_acute_all_events = df_acute_all_events.withColumn(
                                        "Diagnosis_type",
                                    F.when(
                                        F.col("Event_Type") == "Diagnosis_ACUTE",
                                        F.when(F.regexp_extract(F.col("Event_Code"), r'^[A-Z][0-9][0-9A-Z]?$|^[A-Z][0-9][0-9A-Z]+$', 0) != "", "ICD10")
                                    .when(F.regexp_extract(F.col("Event_Code"), r'^[0-9]{6,18}$', 0) != "", "SNOMED")
                                    .otherwise("Other")
                                    ).otherwise("")
    )

    df_acute_all_events = df_acute_all_events.withColumn(
        "Diagnosis_SUS_icd10_3_char",
        F.when(
            (F.col("Diagnosis_type") == "ICD10"),
            F.substring(F.col("Event_Code"), 1, 3)
        ).otherwise("")
        )

    df_acute_all_events = df_acute_all_events.join(df_icd10_ref_cat_3.select(["Alt_Code_3_char", "Description"]),
                                            df_acute_all_events.Diagnosis_SUS_icd10_3_char == df_icd10_ref_cat_3.Alt_Code_3_char,
                                            "left")

    df_acute_all_events = df_acute_all_events.withColumnRenamed("Description", "ICD10_description")

    df_acute_all_events = df_acute_all_events.withColumn(
                                        "Procedure_type",
                                    F.when(
                                        F.col("Event_Type") == "Procedure_ACUTE",
                                        F.when(F.regexp_extract(F.col("Event_Code"), r'^[A-Z][0-9][0-9A-Z]?$|^[A-Z][0-9][0-9A-Z]+$', 0) != "", "OPCS")
                                    .when(F.regexp_extract(F.col("Event_Code"), r'^[0-9]{6,18}$', 0) != "", "SNOMED")
                                    .otherwise("Other")
                                    ).otherwise("")
    )

    df_acute_all_events = df_acute_all_events.join(df_opcs_ref_latest.select(["Code_Without_Decimal", "Title"]),
                                            (df_acute_all_events.Event_Code == df_opcs_ref_latest.Code_Without_Decimal) & (df_acute_all_events.Procedure_type == "OPCS"),
                                            "left")

    df_acute_all_events = df_acute_all_events.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                            (df_acute_all_events.Event_Code == df_chosen_snomed_description.Concept_ID) & ((df_acute_all_events.Diagnosis_type == "SNOMED") | ((df_acute_all_events.Procedure_type == "SNOMED"))),
                                            "left")

    df_acute_all_events = df_acute_all_events.join(
        df_hrg_latest.select(["HRG_Code", "HRG_Name"]),
        (df_acute_all_events.Event_Code == df_hrg_latest.HRG_Code) &
        (df_acute_all_events.Event_Type=="HRG_ACUTE"),
        "left"
    )

    df_acute_all_events = df_acute_all_events.withColumn("description", 
                                    F.when(
                                        (F.col("Event_Type") == "Diagnosis_ACUTE") & (F.col("Diagnosis_type") == "ICD10") & (F.col("ICD10_Description").isNotNull()),
                                        F.concat_ws("",F.lit("Diagnosis: "), F.col("ICD10_Description"), F.lit(" | ") ,F.lit("Type: "), F.col("Record_Classification"))
                                    ).when(
                                        (F.col("Event_Type") == "Procedure_ACUTE") & (F.col("Procedure_type") == "OPCS") & (F.col("Title").isNotNull()),
                                        F.concat_ws("",F.lit("Procedure: "), F.col("Title"), F.lit(" | ") ,  F.lit("Type: "), F.col("Record_Classification"))
                                    ).when(
                                        (F.col("Event_Type") == "Diagnosis_ACUTE") & (F.col("Diagnosis_type") == "SNOMED") & (F.col("Term").isNotNull()),
                                        F.concat_ws("",F.lit("Diagnosis: "), F.col("Term"), F.lit(" | ") ,  F.lit("Type: "), F.col("Record_Classification"))
                                    ).when(
                                        (F.col("Event_Type") == "Procedure_ACUTE") & (F.col("Procedure_type") == "SNOMED") & (F.col("Term").isNotNull()),
                                        F.concat_ws("",F.lit("Procedure: "), F.col("Term"), F.lit(" | ") ,  F.lit("Type: "), F.col("Record_Classification"))
                                    ).when(
                                        (F.col("Event_Type") == "HRG_ACUTE") & (F.col("HRG_Name").isNotNull()),
                                        F.concat_ws("",F.lit("HRG: "), F.col("HRG_Name"), F.lit(" | ") , F.lit("Type: "), F.col("Record_Classification"))
                                    )                           
                                    .otherwise(F.lit(""))  # or F.lit('') if you prefer empty string
    )

    df_sus_acute_all_events = df_acute_all_events.unionByName(df_sus_all_events, allowMissingColumns=True)

    df_sus_acute_all_events = df_sus_acute_all_events.dropDuplicates(["Patient_ID","date","Diagnosis_type", "Procedure_type", "HRG_Name", "Event_Code"])

    df_OoH = read_parquet_file(containerName =config.containerName_bronze, 
                            lakeName=config.lakeName,
                            filePath=config.filePath_OoH)

    df_OoH = df_OoH.withColumn("dataset", F.lit("UC_OoH_All"))
    df_OoH = df_OoH.withColumn("date", F.col("Attendance_Date"))

    # remove duplicates
    df_OoH = df_OoH.distinct()

    df_OoH = df_OoH.withColumn("description", 
                            F.concat_ws("",F.lit("Contact Type: "), F.col("Dimention_1"), F.lit(" | ") , F.lit("Outcome: "), F.col("Dimention_2"))
    ) 

    df_999 = read_parquet_file(containerName =config.containerName_bronze, 
                            lakeName=config.lakeName,
                            filePath=config.filePath_999)

    df_999 = df_999.withColumn("dataset", F.lit("UC_999_All"))
    df_999 = df_999.withColumn("date", F.col("Attendance_Date"))

    df_999 = df_999.withColumn("description", 
                            F.concat_ws("",F.lit("Chief Complaint: "), F.col("Dimention_1"), F.lit(" | ") , F.lit("Response: "), F.col("Dimention_2"))
    ) 

    # drop duplicates
    df_999 = df_999.dropDuplicates(["Patient_ID","date","Dimention_1", "Dimention_2"])

    df_111 = read_parquet_file(containerName =config.containerName_bronze, 
                            lakeName=config.lakeName,
                            filePath=config.filePath_111)

    df_111_reference = read_csv_file(containerName =config.containerName_platinum, 
                                        lakeName= config.lakeName,
                                        filePath= config.filePath_111_mapping)

    df_111 = df_111.withColumn("dataset", F.lit("UC_111_All"))
    df_111 = df_111.withColumn("date", F.col("Attendance_Date"))

    df_111_data = processing.process_111_dataset(df_111, df_111_reference)

    df_111_data = df_111_data.withColumn("description", 
                            F.concat_ws("",F.lit("Symptom: "), F.col("SG_Description"),
                                        F.lit(" | ") , F.lit("SD_Description: "), F.col("SD_Description"),
                                        F.lit(" | ") , F.lit("DX_Description: "), F.col("DX_Description"))
    ) 

    df_111_data = df_111_data.withColumn("SG_Description", 
                                        F.when(F.col("SG_Description").isNull(), "Unknown")
                                        .otherwise(F.col("SG_Description"))
                                        )

    df_111_data = df_111_data.dropDuplicates(["Patient_ID","date","SG_Description", "SD_Description", "DX_Description"])

    i = 0
    for ecds_filepath in config.filePath_ecds:
        
        if i==0:
            df_ecds_data_full =  read_parquet_file(containerName=config.containerName_bronze,
                                            lakeName=config.lakeName,
                                            filePath=ecds_filepath)
            
        else:
            df = read_parquet_file(containerName=config.containerName_bronze,
                                lakeName=config.lakeName,
                                filePath=ecds_filepath)
        
            df_ecds_data_full = df_ecds_data_full.unionByName(df, allowMissingColumns=True)
        
        print("Imported ", ecds_filepath)

        i=i+1

    df_ecds_data_full = df_ecds_data_full.withColumn("dataset", F.lit("ECDS"))
    df_ecds_data_joined_acute = df_ecds_data_full.join(df_acute.drop("dataset"), df_acute.RecID == df_ecds_data_full.Generated_Record_ID, how="inner")
    df_ecds_data_joined_sus = df_ecds_data_full.join(df_sus_all.drop("dataset", "Emergency_Care_Attendance_Source_Snomed_CT"), on="Spell_ID", how="inner")
    df_ecds_data = df_ecds_data_joined_acute.unionByName(df_ecds_data_joined_sus, allowMissingColumns=True)
    df_ecds_data = df_ecds_data.dropDuplicates(subset=["Patient_ID", "date","Emergency_Care_Chief_Complaint_Snomed_CT"])

    df_ecds_data_chief_complaint = df_ecds_data.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                                                                        df_ecds_data.Emergency_Care_Chief_Complaint_Snomed_CT == df_chosen_snomed_description.Concept_ID,
                                                                                        how="inner").withColumnRenamed("Term", "Emergency_Care_Chief_Complaint").drop("Term", "Concept_ID")

    df_ecds_data_chief_complaint = df_ecds_data_chief_complaint.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                                                                        df_ecds_data_chief_complaint.Emergency_Care_Acuity_Snomed_CT == df_chosen_snomed_description.Concept_ID,
                                                                                        how="left").withColumnRenamed("Term", "Emergency_Care_Acuity").drop("Term", "Concept_ID")

    df_ecds_data_chief_complaint = df_ecds_data_chief_complaint.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                                                                        df_ecds_data_chief_complaint.Emergency_Care_Chief_Complaint_Extended == df_chosen_snomed_description.Concept_ID,
                                                                                        how="left").withColumnRenamed("Term", "Emergency_Care_Chief_Complaint_Extended_Term").drop("Term", "Concept_ID")

    df_ecds_data_chief_complaint = df_ecds_data_chief_complaint.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                                                                        df_ecds_data_chief_complaint.Emergency_Care_Discharge_Status_Snomed_CT == df_chosen_snomed_description.Concept_ID,
                                                                                        how="left").withColumnRenamed("Term", "Emergency_Care_Discharge_Status").drop("Term", "Concept_ID")

    df_ecds_data_chief_complaint = df_ecds_data_chief_complaint.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                                                                        df_ecds_data_chief_complaint.Emergency_Care_Discharge_Follow_Up_Snomed_CT == df_chosen_snomed_description.Concept_ID,
                                                                                        how="left").withColumnRenamed("Term", "Emergency_Care_Discharge_Follow_Up").drop("Term", "Concept_ID")

    df_ecds_data_chief_complaint = df_ecds_data_chief_complaint.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                                                                        df_ecds_data_chief_complaint.Primary_Diagnosis == df_chosen_snomed_description.Concept_ID,
                                                                                        how="left").withColumnRenamed("Term", "Primary_Diagnosis_ECDS").drop("Term", "Concept_ID")

    df_ecds_data_chief_complaint = df_ecds_data_chief_complaint.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                                                                        df_ecds_data_chief_complaint.Primary_Procedure_Code == df_chosen_snomed_description.Concept_ID,
                                                                                        how="left").withColumnRenamed("Term", "Primary_Procedure_Code_ECDS").drop("Term", "Concept_ID")

    df_ecds_data_chief_complaint = df_ecds_data_chief_complaint.withColumn("description", 
                                                                        F.concat_ws("",F.lit("Chief_Complaint: "), F.col("Emergency_Care_Chief_Complaint"),
                                                                                    F.lit(" | ") , F.lit("Complaint Extended Term: "), F.col("Emergency_Care_Chief_Complaint_Extended_Term"),
                                                                                    F.lit(" | ") , F.lit("Acuity: "), F.col("Emergency_Care_Acuity"),
                                                                                    F.lit(" | ") , F.lit("Primary Diagnosis: "), F.col("Primary_Diagnosis_ECDS"),
                                                                                    F.lit(" | ") , F.lit("Procedure: "), F.col("Primary_Procedure_Code_ECDS"),
                                                                                    F.lit(" | ") , F.lit("Discharge_Status: "), F.col("Emergency_Care_Discharge_Status"),
                                                                                    F.lit(" | ") , F.lit("Discharge_Follow_Up: "), F.col("Emergency_Care_Discharge_Follow_Up"))
                                                                        )


    df_gp_appointments = read_parquet_file(containerName =config.containerName_bronze, 
                                        lakeName=config.lakeName,
                                        filePath=config.filePath_gp_all)

    df_gp_appointments = df_gp_appointments.withColumn("dataset", F.lit("GP_Appointments"))
    df_gp_appointments = df_gp_appointments.withColumn("date", F.col("Attendance_Date"))
    df_gp_appointments = df_gp_appointments.withColumn("description", F.col("Record_Classification"))

    df_gp_meds = read_parquet_file(containerName =config.containerName_bronze, 
                                        lakeName=config.lakeName,
                                        filePath=config.filePath_gp_meds)

    df_gp_meds = df_gp_meds.withColumn("dataset", F.lit("GPMeds_All"))
    df_gp_meds = df_gp_meds.withColumn("date", F.col("Attendance_Date"))
    df_gp_meds = df_gp_meds.withColumn("SnomedCode", F.col("Dimention_1"))

    # add descritpion for snomeds and add snomed code column

    df_gp_meds = df_gp_meds.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                    df_gp_meds.SnomedCode == df_chosen_snomed_description.Concept_ID,
                                    how="left" )

    df_gp_meds = df_gp_meds.filter(F.col("SnomedCode") != "-1") # remove where code is -1

    # need to convert to match other tables

    df_gp_meds = df_gp_meds.withColumn("Dimention_4", F.col("Dimention_4").cast("string"))         

    df_gp_meds = df_gp_meds.withColumn("description", F.col("Term"))

    df_all_activity = df_gp_events.unionByName(df_sus_acute_all_events,allowMissingColumns=True)\
                                .unionByName(df_OoH, allowMissingColumns=True)\
                                .unionByName(df_999.drop("Decision_to_Refer_to_Service_Date", "Discharge_Date"), allowMissingColumns=True)\
                                .unionByName(df_111_data.drop("Decision_to_Refer_to_Service_Date", "Discharge_Date"), allowMissingColumns=True)\
                                .unionByName(df_cancer_site.drop("diagnosis_date_earliest"), allowMissingColumns=True)\
                                .unionByName(df_ecds_data_chief_complaint, allowMissingColumns=True)\
                                .unionByName(df_gp_meds, allowMissingColumns=True)\
                                .unionByName(df_gp_appointments.drop("Decision_to_Refer_to_Service_Date", "Discharge_Date"), allowMissingColumns=True)

    fullPath_output=""

    df_all_activity.write.mode("overwrite").parquet(
        fullPath_output,
    )

else:
    df_all_activity = read_parquet_file(containerName =config.containerName_platinum, 
                                     lakeName=config.lakeName,
                                     filePath= f"")

# COMMAND ----------

# MAGIC %md
# MAGIC # Filter activity table

# COMMAND ----------

# MAGIC %md
# MAGIC - get activity of non-cancer patients
# MAGIC - Filter activity table by date
# MAGIC - filter activity table by patients (those with complete data)
# MAGIC - filter to patients over 40 in period 2023 onwards
# MAGIC - identify future cancer diagnosis, join to event
# MAGIC - remove cancer cases with incomplete data
# MAGIC - keep cases when cancer occurred 1yr after event
# MAGIC - use activity in 2023
# MAGIC - observe cancer in 2023 and 2024
# MAGIC

# COMMAND ----------

# get activity of cancer patients
# identify future cancer diagnosis, join to event

df_all_activity = df_all_activity.drop("Age")

df_all_activity = df_all_activity.withColumn("dataset",
                                             F.when(F.col("dataset")=="SUS_Activity_Extract", "Acute_All").otherwise(F.col("dataset")))

df_cancer_patient_activity = df_all_activity.join(df_cancer_site_complete_data_no_ltc_flag.select(["Patient_ID", "diagnosis_date_earliest", "Age"]),
                                                  on = "Patient_ID",
                                                  how="inner")

df_cancer_patient_activity = df_cancer_patient_activity.withColumn("days_between_activity_diagnosis", F.datediff(F.col("diagnosis_date_earliest"), F.col("date")))

# remove activity of cancer patients after their diagnosis
df_cancer_patient_activity = df_cancer_patient_activity.filter(F.col("days_between_activity_diagnosis")>=0)

# remove the lung cancer patients from the whole population
df_patients_with_complete_data_non_cancer = df_patients_with_complete_data.join(df_cancer_site_complete_data_no_ltc_flag.select("Patient_ID"), on = "Patient_ID",
                                                                                how="leftanti")

df_non_cancer_patient_activity = df_all_activity.join(df_patients_with_complete_data_non_cancer.select(["Patient_ID"] + list_cohort_ages),
                                                      on = "Patient_ID",
                                                      how="inner")


df_non_cancer_patient_activity = df_non_cancer_patient_activity.withColumn("Age", F.least(*list_cohort_ages))

df_combined_activity = df_cancer_patient_activity.unionByName(df_non_cancer_patient_activity, allowMissingColumns=True)

df_combined_activity = df_combined_activity.withColumn(f"cancer_in_next_{days_ppv}_days", 
                                           F.when((F.col("days_between_activity_diagnosis")<=days_ppv) & 
                                                  (F.col("days_between_activity_diagnosis")>=0),1).otherwise(0))

df_combined_activity = df_combined_activity.join(df_smoking_flags, on = "Patient_ID", how="left")


# only consider activty up to the latest_cancer_date
df_combined_activity = df_combined_activity.filter((F.col("date")<latest_cancer_date))



# COMMAND ----------

df_combined_activity.groupby(f"cancer_in_next_{days_ppv}_days").agg(F.countDistinct("Patient_ID")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Join mapping files to create flags

# COMMAND ----------

# GP
df_gp_red_flags = df_combined_activity.join(df_gp_red_flags_ref.drop("term"),
                                           df_combined_activity.Concept_ID == df_gp_red_flags_ref.code, 'inner')


# Add thrombocytosis via high platelet count
df_patients_with_platelet_count = df_combined_activity.filter(F.col("description")=="Platelet count")

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

df_gp_amber_flags = df_combined_activity.join(df_gp_amber_flags_ref.drop("term"),
                                          df_combined_activity.Concept_ID == df_gp_amber_flags_ref.code,
                                                                    'inner')

df_prescription_flags = df_combined_activity.join(df_medication_mapping,
                                                         df_combined_activity.Concept_ID == df_medication_mapping.code,
                                                         'inner')

# COMMAND ----------

df_111_red_flags = df_combined_activity.join(df_111_mapping.filter(F.col("grouping")=="red_flag"), df_combined_activity.SG_Description == df_111_mapping.symptom_111, 'inner')

df_111_red_flags = df_111_red_flags.withColumn("grouping", F.lit("red_flag"))
df_111_red_flags = df_111_red_flags.withColumn("symptom_name", F.col("NICE_matching"))

df_111_amber_flags = df_combined_activity.join(df_111_mapping.filter(F.col("grouping")=="amber_flag"), df_combined_activity.SG_Description == df_111_mapping.symptom_111, 'inner')

df_111_amber_flags = df_111_amber_flags.withColumn("grouping", F.lit("amber_flag"))
df_111_amber_flags = df_111_amber_flags.withColumn("symptom_name", F.col("NICE_matching"))

df_ecds_red_flags = df_combined_activity.join(df_ecds_mapping_red_flag,
                                             df_combined_activity.Emergency_Care_Chief_Complaint_Snomed_CT == df_ecds_mapping_red_flag.SNOMED_Code, 'inner')

df_ecds_red_flags = df_ecds_red_flags.withColumn("grouping", F.lit("red_flag"))
df_ecds_red_flags = df_ecds_red_flags.withColumn("symptom_name", F.col("NICE_matching"))

df_ecds_amber_flags = df_combined_activity.join(df_ecds_mapping_amber_flag,
                                               df_combined_activity.Emergency_Care_Chief_Complaint_Snomed_CT == df_ecds_mapping_amber_flag.SNOMED_Code, 'inner')

df_ecds_amber_flags = df_ecds_amber_flags.withColumn("grouping", F.lit("amber_flag"))
df_ecds_amber_flags = df_ecds_amber_flags.withColumn("symptom_name", F.col("NICE_matching"))


# COMMAND ----------

df_acute_red_flag = df_combined_activity.join(df_icd10_codes_red_flag.select(["ICD10", "NICE_matching"]),
                                                                                        (
                                                                                            (df_combined_activity.Event_Code == df_icd10_codes_red_flag.ICD10)
                                                                                            |
                                                                                            (df_combined_activity.Diagnosis_SUS_icd10_3_char == df_icd10_codes_red_flag.ICD10)
                                                                                            ),
                                                                                        'inner')

df_acute_red_flag = df_acute_red_flag.withColumn("grouping", F.lit("red_flag"))
df_acute_red_flag = df_acute_red_flag.withColumn("symptom_name", F.col("NICE_matching"))

df_acute_amber_flag = df_combined_activity.join(df_icd10_codes_amber_flag.select(["ICD10", "NICE_matching"]),
                                                                                        (
                                                                                            (df_combined_activity.Event_Code == df_icd10_codes_amber_flag.ICD10)
                                                                                            |
                                                                                            (df_combined_activity.Diagnosis_SUS_icd10_3_char == df_icd10_codes_amber_flag.ICD10)
                                                                                            ),
                                                                                        'inner')


df_acute_amber_flag = df_acute_amber_flag.withColumn("grouping", F.lit("amber_flag"))
df_acute_amber_flag = df_acute_amber_flag.withColumn("symptom_name", F.col("NICE_matching"))

# COMMAND ----------

df_chest_xray_snomed = df_combined_activity.join(df_mapping_chest_xray.select(["code"]),
                                                     df_combined_activity.Concept_ID == df_mapping_chest_xray.code,
                                                     'inner')

df_chest_xray_snomed = df_chest_xray_snomed.withColumn("symptom_name", F.lit("chest_xray"))

#df_chest_xray_snomed = df_chest_xray_snomed.select(["Patient_ID", "date", "days_between_activity_diagnosis", "description", "symptom_name"])

df_chest_xray_icd10=df_combined_activity.filter((F.col("Diagnosis_type")=="ICD10") 
                                                           & F.col("Event_Code").startswith(config_pathways.cancer_site_mappings[cancer_site]["abnormal_xray_icd10"]))

df_chest_xray_icd10=df_chest_xray_icd10.withColumn("symptom_name", F.lit("chest_xray"))

#df_chest_xray_icd10 = df_chest_xray_icd10.select(["Patient_ID", "date", "days_between_activity_diagnosis", "description", "symptom_name"])

df_chest_xray_procedure = df_combined_activity.filter((F.col("Procedure_type")=="OPCS") 
                                                           & F.col("Event_Code").startswith("U07"))


df_chest_xray_procedure = df_chest_xray_procedure.withColumn("symptom_name", F.lit("chest_xray"))

#df_chest_xray_procedure = df_chest_xray_procedure.select(["Patient_ID", "date", "days_between_activity_diagnosis","description", "symptom_name"])

df_chest_xray_combined = df_chest_xray_snomed.unionByName(df_chest_xray_icd10, allowMissingColumns = True)
df_chest_xray_combined = df_chest_xray_combined.unionByName(df_chest_xray_procedure, allowMissingColumns = True)
df_chest_xray_combined = df_chest_xray_combined.dropDuplicates(subset = ["Patient_ID", "date"])

# COMMAND ----------

common_columns = ["Patient_ID", "date", f"cancer_in_next_{days_ppv}_days", "dataset", "symptom_name", "diagnosis_date_earliest", "days_between_activity_diagnosis", "Smoking_Flag", "Age"]

# COMMAND ----------

df_all_flags = df_gp_red_flags.select(common_columns).unionByName(df_gp_amber_flags.select(common_columns), allowMissingColumns=True)
df_all_flags = df_all_flags.select(common_columns).unionByName(df_prescription_flags.select(common_columns), allowMissingColumns=True)
df_all_flags = df_all_flags.select(common_columns).unionByName(df_111_red_flags.select(common_columns), allowMissingColumns=True)
df_all_flags = df_all_flags.select(common_columns).unionByName(df_111_amber_flags.select(common_columns), allowMissingColumns=True)
df_all_flags = df_all_flags.select(common_columns).unionByName(df_ecds_red_flags.select(common_columns), allowMissingColumns=True)
df_all_flags = df_all_flags.select(common_columns).unionByName(df_ecds_amber_flags.select(common_columns), allowMissingColumns=True)
df_all_flags = df_all_flags.select(common_columns).unionByName(df_acute_red_flag.select(common_columns), allowMissingColumns=True)
df_all_flags = df_all_flags.select(common_columns).unionByName(df_acute_amber_flag.select(common_columns), allowMissingColumns=True)
df_all_flags = df_all_flags.select(common_columns).unionByName(df_chest_xray_combined.select(common_columns), allowMissingColumns=True)

# COMMAND ----------

df_all_flags.groupby("symptom_name").count().display()

# COMMAND ----------

if write_output == True:

    fullPath_output="abfss://"+config.containerName_platinum+"@"+config.lakeName + f""

    df_all_flags.write.mode("overwrite").parquet(
        fullPath_output,
    )

    fullPath_output=""

    df_cancer_site_complete_data_no_ltc_flag.write.mode("overwrite").parquet(
    fullPath_output,
    )


# COMMAND ----------

# MAGIC %md
# MAGIC # Read dataset of all flags

# COMMAND ----------

df_all_flags =  read_parquet_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= f"")

df_all_flags_pd = df_all_flags.toPandas()

# activity between earliest cancer date, and latest activity date (1 year before latest cancer diagnosis)
df_flags_after_earliest_cancer_date = df_all_flags_pd[(df_all_flags_pd["date"]>= earliest_cancer_date) & 
                                                      (df_all_flags_pd["date"]<latest_activity_date)]

df_cancer_site_complete_data_no_ltc_flag = read_parquet_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= f"")

df_cancer_site_complete_data_no_ltc_flag_pd = df_cancer_site_complete_data_no_ltc_flag.toPandas()

# COMMAND ----------

number_cancers = df_cancer_site_complete_data_no_ltc_flag_pd["Patient_ID"].nunique()
print(number_cancers)

# COMMAND ----------

# consider events in the 1 year before diagnosis
df_cancer_patient_flags_in_1_year_before_diagnosis = df_all_flags_pd[(~df_all_flags_pd["diagnosis_date_earliest"].isnull())]

df_cancer_patient_flags_in_1_year_before_diagnosis = df_cancer_patient_flags_in_1_year_before_diagnosis[(df_cancer_patient_flags_in_1_year_before_diagnosis["days_between_activity_diagnosis"]<=365) & (df_cancer_patient_flags_in_1_year_before_diagnosis["days_between_activity_diagnosis"]>=0)]

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate PPV - per dataset

# COMMAND ----------

df_all_flags_counts = df_flags_after_earliest_cancer_date.groupby(["symptom_name", "dataset",f"cancer_in_next_{days_ppv}_days"])["Patient_ID"].count().to_frame()
df_all_flags_counts.columns = ["count"]
df_all_flags_counts = df_all_flags_counts.reset_index()
df_all_flags_counts["symptom_dataset"] = df_all_flags_counts["symptom_name"] + df_all_flags_counts["dataset"]

for dataset in df_all_flags_counts["dataset"].unique():
    print(dataset)
    df = df_all_flags_counts[df_all_flags_counts["dataset"]==dataset]
    df_wide = df.pivot_table(index="symptom_name", columns = f"cancer_in_next_{days_ppv}_days", values="count")
    df_wide = df_wide.fillna(0)
    df_wide["total"] = df_wide[0] + df_wide[1]
    df_wide["ppv (%)"] = 100*df_wide[1]/df_wide["total"]
    df_wide = df_wide.reset_index()
    display(df_wide)


# COMMAND ----------

# drop duplicates so only one symptom per patient
df_single_symptom_all_sources_ppv = df_flags_after_earliest_cancer_date.drop_duplicates(subset=["Patient_ID", "symptom_name"]).groupby(["symptom_name"])[f"cancer_in_next_{days_ppv}_days"].value_counts().to_frame()
df_single_symptom_all_sources_ppv.columns = ["sum"]
df_single_symptom_all_sources_ppv = df_single_symptom_all_sources_ppv.reset_index()
df_all_flags_counts_wide = df_single_symptom_all_sources_ppv.pivot_table(index="symptom_name", columns = f"cancer_in_next_{days_ppv}_days", values="sum")
df_all_flags_counts_wide = df_all_flags_counts_wide.fillna(0)
df_all_flags_counts_wide["total"] = df_all_flags_counts_wide[0] + df_all_flags_counts_wide[1]
df_all_flags_counts_wide["ppv (%)"] = 100*df_all_flags_counts_wide[1]/df_all_flags_counts_wide["total"]
df_all_flags_counts_wide



# COMMAND ----------

# MAGIC %md
# MAGIC # Sensitivity analysis - per symptom

# COMMAND ----------

# calculate sensitivity per feature
# for each feature, what percentage of cancer patients had the symptom prior to their diagnosis 
# only look at cancers with at least 1 year of history available

# COMMAND ----------

# of all cancer patients 
df_num_per_symptom = df_cancer_patient_flags_in_1_year_before_diagnosis.groupby("symptom_name")["Patient_ID"].nunique().to_frame()
df_num_per_symptom.columns = ["count_unique_ids"]
df_num_per_symptom["Sensitivity"] =  100*df_num_per_symptom["count_unique_ids"]/number_cancers
df_num_per_symptom

# COMMAND ----------

# of the patients who had at least one flag of some sort
df_num_per_symptom = df_cancer_patient_flags_in_1_year_before_diagnosis.groupby("symptom_name")["Patient_ID"].nunique().to_frame()
df_num_per_symptom.columns = ["count_unique_ids"]
df_num_per_symptom["Sensitivity"] =  100*df_num_per_symptom["count_unique_ids"]/df_cancer_patient_flags_in_1_year_before_diagnosis["Patient_ID"].nunique()
df_num_per_symptom

# COMMAND ----------

# MAGIC %md
# MAGIC # PPV two symptoms

# COMMAND ----------

def calculate_ppv_for_symptoms(df_activity_for_ppv, s1, s2, target_col):
    """
    Calculate the positive predictive value (PPV) for one or two symptoms.

    For a single symptom (s1 == s2), selects the latest occurrence per patient and uses the target column as is.
    For two symptoms, selects the latest occurrence of each symptom per patient, merges on Patient_ID, and sets the target column to 0 if the values differ (meaning cancer did not occur in the target period for one of the two symptoms), otherwise uses the value of the target column.

    Args:
        df_activity_for_ppv (pd.DataFrame): DataFrame containing activity data for PPV calculation.
        s1 (str): First symptom name.
        s2 (str): Second symptom name.
        target_col (str): Name of the target column indicating cancer outcome.

    Returns:
        pd.DataFrame: DataFrame with counts of cancer outcomes for the symptom pair.
    """
    if s1==s2:
        df_s1_s2 = df_activity_for_ppv[df_activity_for_ppv["symptom_name"] == s1].copy()
        df_s1_s2 = df_s1_s2.sort_values("date").groupby("Patient_ID").tail(1)
        df_s1_s2[target_col+"_final"] = df_s1_s2[target_col]

    else:
        df_s1 = df_activity_for_ppv[df_activity_for_ppv["symptom_name"] == s1]
        df_s1 = df_s1.sort_values("date").groupby("Patient_ID").tail(1)

        df_s2 = df_activity_for_ppv[df_activity_for_ppv["symptom_name"] == s2]
        df_s2 = df_s2.sort_values("date").groupby("Patient_ID").tail(1)

        df_s1_s2 = df_s1.merge(df_s2, on = "Patient_ID", how="inner", suffixes = ("_S1", "_S2"))
        df_s1_s2[target_col+"_final"] = np.where(df_s1_s2[target_col+"_S1"]!=df_s1_s2[target_col+"_S2"], 0,df_s1_s2[target_col+"_S1"])

    df_s1_s2["S1"] = s1
    df_s1_s2["S2"] = s2
    df_s1_s2["S1_S2"] =  df_s1_s2["S1"] + " & " + df_s1_s2["S2"] 

    # cancer in next period only if it applies to both symptoms
    df_result = df_s1_s2.groupby(["S1_S2"])[target_col+"_final"].value_counts().to_frame()
    df_result.columns = ["count"]
    df_result = df_result.reset_index()

    return df_result

def calculate_sensitivity_for_symptoms(df_activitity_for_sensitivity, s1, s2):
    """
    Calculate the sensitivity for one or two symptoms.

    For each symptom, selects the earliest occurrence per patient within the activity table (already limited to one year before diagnosis).
    For two symptoms, merges on Patient_ID to find patients with both symptoms, and calculates the average number of days between activity and diagnosis.

    Args:
        df_activitity_for_sensitivity (pd.DataFrame): DataFrame containing activity data for sensitivity calculation.
        s1 (str): First symptom name.
        s2 (str): Second symptom name.

    Returns:
        pd.DataFrame: DataFrame with count of patients and average days for the symptom pair.
    """
    df_s1 = df_activitity_for_sensitivity[df_activitity_for_sensitivity["symptom_name"] == s1]
    df_s2 = df_activitity_for_sensitivity[df_activitity_for_sensitivity["symptom_name"] == s2]

    # identify earliest date for each symptom - the activity table is already limited to activity one year before diagnosis
    df_s1 = df_s1.sort_values("date").groupby("Patient_ID").head(1)
    df_s2 = df_s2.sort_values("date").groupby("Patient_ID").head(1)

    df_s1_s2 = df_s1.merge(df_s2[["Patient_ID", "days_between_activity_diagnosis"]], on = "Patient_ID", how="inner", suffixes = ("_S1", "_S2"))
    df_s1_s2["S1"] = s1
    df_s1_s2["S2"] = s2
    df_s1_s2["S1_S2"] =  df_s1_s2["S1"] + " & " + df_s1_s2["S2"] 

    df_s1_s2["avg_number_of_days_for_two_symptoms"] = df_s1_s2[["days_between_activity_diagnosis_S1", "days_between_activity_diagnosis_S2"]].mean(axis=1)

    df_result_num_patients = df_s1_s2.groupby(["S1_S2"])["Patient_ID"].nunique().to_frame()
    df_result_num_patients.columns = ["count"]
    df_result_num_patients = df_result_num_patients.reset_index()

    df_result_mean_time = df_s1_s2.groupby(["S1_S2"])["avg_number_of_days_for_two_symptoms"].mean().to_frame().reset_index()

    df_result = df_result_num_patients.merge(df_result_mean_time, on = "S1_S2")

    return df_result

def calculate_ppv_sensitivity_from_activity(df_activity_for_ppv,
                                            df_activitity_for_sensitivity,
                                            list_symptoms,
                                            number_cancers,
                                            target_col = f"cancer_in_next_{days_ppv}_days"):
    """
    Calculate PPV and sensitivity for all pairs of symptoms in the provided list.

    For each pair of symptoms, calculates PPV using calculate_ppv_for_symptoms and sensitivity using calculate_sensitivity_for_symptoms.
    Combines the results into a single DataFrame.

    Args:
        df_activity_for_ppv (pd.DataFrame): DataFrame for PPV calculation.
        df_activitity_for_sensitivity (pd.DataFrame): DataFrame for sensitivity calculation.
        list_symptoms (list): List of symptom names to consider.
        number_cancers (int): Total number of cancer patients for sensitivity denominator.
        target_col (str, optional): Name of the target column indicating cancer outcome.

    Returns:
        pd.DataFrame: Combined DataFrame with PPV, sensitivity, and average days for each symptom pair.
    """
    # get the latest reported per patient
    # this is to prevent multiple counting of the same symptom (e.g. if same symptom reported in multiple places)
    list_symptoms.sort()
    i = 0
    j = 0

    df_all_results_ppv = pd.DataFrame()
    df_all_results_sensitivity = pd.DataFrame()

    for i in range(0,len(list_symptoms)):
        for j in range(i,len(list_symptoms)):
            s1 = list_symptoms[i]
            s2 = list_symptoms[j]
            
            print("S1: ", s1, "S2: ", s2)

            df_result_ppv = calculate_ppv_for_symptoms(df_activity_for_ppv, s1, s2, target_col)
            df_all_results_ppv = pd.concat([df_all_results_ppv, df_result_ppv])

            df_result_sensitivity = calculate_sensitivity_for_symptoms(df_activitity_for_sensitivity, s1, s2)
            df_all_results_sensitivity = pd.concat([df_all_results_sensitivity, df_result_sensitivity])

            j=j+1
        
        i=i+1
    
    df_all_results_ppv = df_all_results_ppv.pivot_table(index = "S1_S2", columns = target_col+"_final", values = "count").reset_index().fillna(0)
    df_all_results_ppv["S1"] = df_all_results_ppv["S1_S2"].apply(lambda x:x.split(" & ")[0])
    df_all_results_ppv["S2"] = df_all_results_ppv["S1_S2"].apply(lambda x:x.split(" & ")[1])

    # if no instances of counts for taget col 0 or 1, create the column
    if 0 not in df_all_results_ppv.columns:
        df_all_results_ppv[0] = 0
    
    if 1 not in df_all_results_ppv.columns:
        df_all_results_ppv[0] = 0

    # fill nulls with 0
    df_all_results_ppv = df_all_results_ppv.fillna(0)
    df_all_results_ppv["total"] = df_all_results_ppv[0] + df_all_results_ppv[1]
    df_all_results_ppv["ppv (%)"] = 100*df_all_results_ppv[1]/df_all_results_ppv["total"]

    df_all_results_sensitivity["sensitivity"] = 100*df_all_results_sensitivity["count"]/number_cancers   

    df_all_results_combined = df_all_results_ppv.merge(df_all_results_sensitivity, on = "S1_S2", how="inner")

    df_all_results_combined["S1_S2"] = np.where(df_all_results_combined["S1"]==df_all_results_combined["S2"], df_all_results_combined["S1"], df_all_results_combined["S1_S2"])

    return df_all_results_combined


# COMMAND ----------

def plot_ppv_sensitivity(df_ppv_sensitivity,min_ppv_for_scatter=1,min_sensitivity_for_scatter=2, xlim_ppv = 4, ylim_sensitivity = 50):
    sns.set(font_scale = 1.5, style="whitegrid")

    df = df_ppv_sensitivity.copy()

    # small number suppression
    df["ppv (%)"] = np.where(df["total"]<=100, np.NaN, df["ppv (%)"])
    df["sensitivity"] = np.where(df["count"]<=10, np.NaN, df["sensitivity"])

    heatmap_data = df.pivot(index="S2", columns="S1", values="ppv (%)")

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("PPV (%) Heatmap for Symptom Pairs")
    plt.xlabel("S2")
    plt.ylabel("S1")
    plt.show()


    heatmap_data = df.pivot(index="S2", columns="S1", values="sensitivity")
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Sensitivity Heatmap for Symptom Pairs")
    plt.show()

    # scatterplot sensitivity vs ppv

    #sns.set(font_scale = 1.5, style="white")
    plt.figure(figsize=(10, 10))

    df = df_ppv_sensitivity[df_ppv_sensitivity["ppv (%)"]>=min_ppv_for_scatter]
    df=df[df["sensitivity"]>=min_sensitivity_for_scatter]

    sns.scatterplot(x="ppv (%)", y ="sensitivity", data = df, hue = "S1_S2", style = "S1_S2", palette= color_map , markers= marker_map, s=150)
    plt.xlabel("PPV (%)")
    plt.ylabel("Sensitivity (%)")

    plt.legend(bbox_to_anchor = [1.05,1], loc = 2)
    plt.xlim([0,xlim_ppv]);
    plt.ylim([0, ylim_sensitivity]);

    plt.axhline(y=min_sensitivity_for_scatter, linestyle="--" )
    plt.axvline(x=min_ppv_for_scatter, linestyle="--")

    # scatterplot ppv vs avg_number_of_days_for_two_symptoms

    #sns.set(font_scale = 1.5, style="white")
    plt.figure(figsize=(10, 10))

    df = df_ppv_sensitivity[df_ppv_sensitivity["ppv (%)"]>=min_ppv_for_scatter]
    df=df[df["sensitivity"]>=min_sensitivity_for_scatter]

    sns.scatterplot(x="ppv (%)", y ="avg_number_of_days_for_two_symptoms", data = df, hue = "S1_S2", style = "S1_S2",palette= color_map , markers= marker_map, s=150)
    plt.xlabel("PPV (%)")
    plt.ylabel("Average number of days between symptoms and diagnosis")
    plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

def identify_val_in_list(col_name, list_of_items, null_return = None):
    for val in list_of_items:
        if val in col_name:
            return val
    return null_return

# COMMAND ----------

# MAGIC %md
# MAGIC ## all patients

# COMMAND ----------

df_ppv_sensitivity = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date,
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis,
                                            list_symptoms = list_of_symptoms_incl_prescriptions,
                                            number_cancers = number_cancers,
                                            target_col = f"cancer_in_next_{days_ppv}_days")

df_ppv_sensitivity

# COMMAND ----------

number_cancers = df_cancer_site_complete_data_no_ltc_flag_pd["Patient_ID"].nunique()

print(number_cancers)

print(df_flags_after_earliest_cancer_date[df_flags_after_earliest_cancer_date[f"cancer_in_next_{days_ppv}_days"]==1]["Patient_ID"].nunique())

print(df_cancer_patient_flags_in_1_year_before_diagnosis["Patient_ID"].nunique())

# COMMAND ----------

# create distinct colour and symbol for each symptom

symptoms = df_ppv_sensitivity["S1_S2"].unique()

# --- 1. Create a large palette of distinct colours ---
# You have ~100 symptoms, so use a large palette.
colors = sns.color_palette("tab20", 20)   # 20-color palette
# Repeat until you have enough colours
color_list = (colors * ((len(symptoms) // 20) + 1))[:len(symptoms)]
color_map = {sym: color_list[i] for i, sym in enumerate(symptoms)}

# --- 2. Create a marker set and cycle through it ---
markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "H", "d"]
marker_list = (markers * ((len(symptoms) // len(markers)) + 1))[:len(symptoms)]
marker_map = {sym: marker_list[i] for i, sym in enumerate(symptoms)}

# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

plot_ppv_sensitivity(df_ppv_sensitivity,min_ppv_for_scatter=min_ppv_for_scatter,min_sensitivity_for_scatter=min_sensitivity_for_scatter)

# COMMAND ----------

# MAGIC %md
# MAGIC ## smokers only

# COMMAND ----------

number_cancers_smokers = df_cancer_site_complete_data_no_ltc_flag_pd[df_cancer_site_complete_data_no_ltc_flag_pd["Smoking_Flag"]==1]["Patient_ID"].nunique()


# COMMAND ----------

df_ppv_sensitivity_smokers = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date[df_flags_after_earliest_cancer_date["Smoking_Flag"]==1],
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["Smoking_Flag"]==1],
                                            list_symptoms = list_of_symptoms_incl_prescriptions,
                                            number_cancers = number_cancers_smokers,
                                            target_col = f"cancer_in_next_{days_ppv}_days")

# COMMAND ----------

print(number_cancers_smokers)

print(df_flags_after_earliest_cancer_date[(df_flags_after_earliest_cancer_date["Smoking_Flag"]==1) & 
                                              (df_flags_after_earliest_cancer_date[f"cancer_in_next_{days_ppv}_days"]==1)]["Patient_ID"].nunique())

print(df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["Smoking_Flag"]==1]["Patient_ID"].nunique())

# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

plot_ppv_sensitivity(df_ppv_sensitivity_smokers,min_ppv_for_scatter=min_ppv_for_scatter,min_sensitivity_for_scatter=min_sensitivity_for_scatter)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Never smoked

# COMMAND ----------

number_cancers_never_smoked = df_cancer_site_complete_data_no_ltc_flag_pd[df_cancer_site_complete_data_no_ltc_flag_pd["Smoking_Flag"]==0]["Patient_ID"].nunique()


# COMMAND ----------

df_ppv_sensitivity_non_smokers = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date[df_flags_after_earliest_cancer_date["Smoking_Flag"]==0],
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["Smoking_Flag"]==0],
                                            list_symptoms = list_of_symptoms_incl_prescriptions,
                                            number_cancers = number_cancers_never_smoked,
                                            target_col = f"cancer_in_next_{days_ppv}_days")
print(number_cancers_never_smoked)

print(df_flags_after_earliest_cancer_date[(df_flags_after_earliest_cancer_date["Smoking_Flag"]==0) & 
                                              (df_flags_after_earliest_cancer_date[f"cancer_in_next_{days_ppv}_days"]==1)]["Patient_ID"].nunique())

print(df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["Smoking_Flag"]==0]["Patient_ID"].nunique())

# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

plot_ppv_sensitivity(df_ppv_sensitivity_non_smokers,min_ppv_for_scatter=min_ppv_for_scatter,min_sensitivity_for_scatter=min_sensitivity_for_scatter)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Age over 74

# COMMAND ----------

number_cancers_over_74 = df_cancer_site_complete_data_no_ltc_flag_pd[df_cancer_site_complete_data_no_ltc_flag_pd["Age"]>74]["Patient_ID"].nunique()


# COMMAND ----------

df_ppv_sensitivity_over_74 = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date[df_flags_after_earliest_cancer_date["Age"]>74],
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["Age"]>74],
                                            list_symptoms = list_of_symptoms_incl_prescriptions,
                                            number_cancers = number_cancers_over_74,
                                            target_col = f"cancer_in_next_{days_ppv}_days")

# COMMAND ----------

print(number_cancers_over_74)

print(df_flags_after_earliest_cancer_date[(df_flags_after_earliest_cancer_date["Age"]>74) & 
                                              (df_flags_after_earliest_cancer_date[f"cancer_in_next_{days_ppv}_days"]==1)]["Patient_ID"].nunique())

print(df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["Age"]>74]["Patient_ID"].nunique())

# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

plot_ppv_sensitivity(df_ppv_sensitivity_over_74,min_ppv_for_scatter=min_ppv_for_scatter,min_sensitivity_for_scatter=min_sensitivity_for_scatter)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Age under 74

# COMMAND ----------

number_cancers_under_74 = df_cancer_site_complete_data_no_ltc_flag_pd[df_cancer_site_complete_data_no_ltc_flag_pd["Age"]<=74]["Patient_ID"].nunique()


# COMMAND ----------

df_ppv_sensitivity_under_74 = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date[df_flags_after_earliest_cancer_date["Age"]<=74],
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["Age"]<=74],
                                            list_symptoms = list_of_symptoms_incl_prescriptions,
                                            number_cancers = number_cancers_under_74,
                                            target_col = f"cancer_in_next_{days_ppv}_days")

# COMMAND ----------

print(number_cancers_under_74)

print(df_flags_after_earliest_cancer_date[(df_flags_after_earliest_cancer_date["Age"]<=74) & 
                                              (df_flags_after_earliest_cancer_date[f"cancer_in_next_{days_ppv}_days"])]["Patient_ID"].nunique())

print(df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["Age"]<=74]["Patient_ID"].nunique())



# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

plot_ppv_sensitivity(df_ppv_sensitivity_under_74,min_ppv_for_scatter=min_ppv_for_scatter,min_sensitivity_for_scatter=min_sensitivity_for_scatter)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GP only

# COMMAND ----------

number_cancers = df_cancer_site_complete_data_no_ltc_flag_pd["Patient_ID"].nunique()

df_ppv_sensitivity_gp_only = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date[df_flags_after_earliest_cancer_date["dataset"]=="gp_events"],
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis[df_cancer_patient_flags_in_1_year_before_diagnosis["dataset"]=="gp_events"],
                                            list_symptoms = list_of_symptoms_incl_prescriptions,
                                            number_cancers = number_cancers,
                                            target_col = f"cancer_in_next_{days_ppv}_days")

df_ppv_sensitivity_gp_only

# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

plot_ppv_sensitivity(df_ppv_sensitivity_gp_only,min_ppv_for_scatter=min_ppv_for_scatter,min_sensitivity_for_scatter=min_sensitivity_for_scatter)

# COMMAND ----------

# MAGIC %md
# MAGIC # Number of combinations of symptom - location

# COMMAND ----------

df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset = df_cancer_patient_flags_in_1_year_before_diagnosis.copy()
df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset["symptom_name"] = df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset["dataset"] + "_" + df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset["symptom_name"] 

df_flags_after_earliest_cancer_date_with_dataset = df_flags_after_earliest_cancer_date.copy()
df_flags_after_earliest_cancer_date_with_dataset["symptom_name"] = df_flags_after_earliest_cancer_date_with_dataset["dataset"] + "_" + df_flags_after_earliest_cancer_date_with_dataset["symptom_name"] 

list_datasets = df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset["dataset"].unique()

list_of_symptoms_incl_prescriptions_by_location = []

for symptom in list_of_symptoms_incl_prescriptions:
    for dataset in list_datasets:
        list_of_symptoms_incl_prescriptions_by_location.append(dataset + "_" + symptom)


list_of_symptoms_incl_prescriptions_by_location.sort()

list_of_symptoms_incl_prescriptions_by_location

# COMMAND ----------

df_ppv_sensitivity_with_dataset = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date_with_dataset,
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset,
                                            list_symptoms = list_of_symptoms_incl_prescriptions_by_location,
                                            number_cancers = number_cancers,
                                            target_col = f"cancer_in_next_{days_ppv}_days")

df_ppv_sensitivity_with_dataset["dataset"] = df_ppv_sensitivity_with_dataset["S1_S2"].apply(lambda col_name : identify_val_in_list(col_name,list_datasets, null_return = "overall"))

df_ppv_sensitivity_with_dataset

# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

sns.set(font_scale = 1, style="whitegrid")

df = df_ppv_sensitivity_with_dataset.copy()

df=df.sort_values(by="dataset")

# small number suppression
df["ppv (%)"] = np.where(df["total"]<=100, np.NaN, df["ppv (%)"])
df["sensitivity"] = np.where(df["count"]<=10, np.NaN, df["sensitivity"])

heatmap_data = df.pivot(index="S2", columns="S1", values="ppv (%)")

plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title("PPV (%) Heatmap for Symptom Pairs")
plt.show()


heatmap_data = df.pivot(index="S2", columns="S1", values="sensitivity")
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title("Sensitivity Heatmap for Symptom Pairs")
plt.show()

# COMMAND ----------


# scatterplot sensitivity vs ppv

df = df_ppv_sensitivity_with_dataset.copy()

sns.set(font_scale = 1.2, style="white")
plt.figure(figsize=(10, 10))

df = df[df["ppv (%)"]>=min_ppv_for_scatter]
df=df[df["sensitivity"]>=min_sensitivity_for_scatter]
#df = df[df["S1"]!=df["S2"]]

sns.scatterplot(x="ppv (%)", y ="sensitivity", data = df, hue = "S1_S2", style = "S1_S2",markers = True, s=100)
plt.xlabel("PPV (%)")
plt.ylabel("Sensitivity (%)")

plt.legend(bbox_to_anchor = [1.05,1], loc = 2)
plt.xlim([0,4]);
plt.ylim([0, 50]);

plt.axhline(y=min_sensitivity_for_scatter, linestyle="--" )
plt.axvline(x=min_ppv_for_scatter, linestyle="--")

# scatterplot ppv vs avg_number_of_days_for_two_symptoms

sns.set(font_scale = 1.2, style="white")
plt.figure(figsize=(10, 10))

sns.scatterplot(x="ppv (%)", y ="avg_number_of_days_for_two_symptoms", data = df, hue = "S1_S2", style = "S1_S2",markers = True, s=100)
plt.xlabel("PPV (%)")
plt.ylabel("Average number of days between symptoms and diagnosis")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### breakdown by route to diagnosis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Emergency

# COMMAND ----------

list_ids_emergency = df_cancer_site_complete_data_no_ltc_flag_pd[df_cancer_site_complete_data_no_ltc_flag_pd["route_earliest"]=="Emergency presentation"]["Patient_ID"].unique()

df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset_with_route = df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset.merge(df_cancer_site_complete_data_no_ltc_flag_pd[["Patient_ID", "route_earliest"]],
                                                                                                                             on = "Patient_ID",
                                                                                                                             how = "left")

df_flags_after_earliest_cancer_date_with_dataset_with_route = df_flags_after_earliest_cancer_date_with_dataset.merge(df_cancer_site_complete_data_no_ltc_flag_pd[["Patient_ID", "route_earliest"]],
                                                                                                                             on = "Patient_ID",
                                                                                                                             how = "left")

# COMMAND ----------

df_ppv_sensitivity_with_dataset_emergency = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date_with_dataset_with_route[
                                                                                                            (df_flags_after_earliest_cancer_date_with_dataset_with_route["cancer_in_next_365_days"]) |                                                                                                                  (df_flags_after_earliest_cancer_date_with_dataset_with_route["cancer_in_next_365_days"]==1 & (df_flags_after_earliest_cancer_date_with_dataset_with_route["route_earliest"]=="Emergency presentation"))],
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset_with_route[df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset_with_route["route_earliest"]=="Emergency presentation"],
                                            list_symptoms = list_of_symptoms_incl_prescriptions_by_location,
                                            number_cancers = len(list_ids_emergency),
                                            target_col = f"cancer_in_next_{days_ppv}_days")

df_ppv_sensitivity_with_dataset_emergency["dataset"] = df_ppv_sensitivity_with_dataset_emergency["S1_S2"].apply(lambda col_name : identify_val_in_list(col_name,list_datasets, null_return = "overall"))

df_ppv_sensitivity_with_dataset_emergency

# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

sns.set(font_scale = 1, style="whitegrid")

df = df_ppv_sensitivity_with_dataset_emergency.copy()

df=df.sort_values(by="dataset")

# small number suppression
df["ppv (%)"] = np.where(df["total"]<=100, np.NaN, df["ppv (%)"])
df["sensitivity"] = np.where(df["count"]<=10, np.NaN, df["sensitivity"])

heatmap_data = df.pivot(index="S2", columns="S1", values="ppv (%)")

plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title("PPV (%) Heatmap for Symptom Pairs")
plt.show()


heatmap_data = df.pivot(index="S2", columns="S1", values="sensitivity")
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title("Sensitivity Heatmap for Symptom Pairs")
plt.show()

# COMMAND ----------


# scatterplot sensitivity vs ppv

df = df_ppv_sensitivity_with_dataset_emergency.copy()

sns.set(font_scale = 1.2, style="white")
plt.figure(figsize=(10, 10))

df = df[df["ppv (%)"]>=min_ppv_for_scatter]
df=df[df["sensitivity"]>=min_sensitivity_for_scatter]
#df = df[df["S1"]!=df["S2"]]

sns.scatterplot(x="ppv (%)", y ="sensitivity", data = df, hue = "S1_S2", style = "S1_S2",markers = True, s=100)
plt.xlabel("PPV (%)")
plt.ylabel("Sensitivity (%)")

plt.legend(bbox_to_anchor = [1.05,1], loc = 2)
plt.xlim([0,4]);
plt.ylim([0, 50]);

plt.axhline(y=min_sensitivity_for_scatter, linestyle="--" )
plt.axvline(x=min_ppv_for_scatter, linestyle="--")

# scatterplot ppv vs avg_number_of_days_for_two_symptoms

sns.set(font_scale = 1.2, style="white")
plt.figure(figsize=(10, 10))

sns.scatterplot(x="ppv (%)", y ="avg_number_of_days_for_two_symptoms", data = df, hue = "S1_S2", style = "S1_S2",markers = True, s=100)
plt.xlabel("PPV (%)")
plt.ylabel("Average number of days between symptoms and diagnosis")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)
plt.ylim([0,365])

# COMMAND ----------

# MAGIC %md
# MAGIC #### GP referral

# COMMAND ----------

# GP REFERRAL
list_ids_gp_referral = df_cancer_site_complete_data_no_ltc_flag_pd[df_cancer_site_complete_data_no_ltc_flag_pd["route_earliest"]=="GP referral"]["Patient_ID"].unique()


df_ppv_sensitivity_with_dataset_gp = calculate_ppv_sensitivity_from_activity(df_activity_for_ppv = df_flags_after_earliest_cancer_date_with_dataset_with_route[
                                                                                                            (df_flags_after_earliest_cancer_date_with_dataset_with_route["cancer_in_next_365_days"]) |                                                                                                                  (df_flags_after_earliest_cancer_date_with_dataset_with_route["cancer_in_next_365_days"]==1 & (df_flags_after_earliest_cancer_date_with_dataset_with_route["route_earliest"]=="GP referral"))],
                                            df_activitity_for_sensitivity = df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset_with_route[df_cancer_patient_flags_in_1_year_before_diagnosis_with_dataset_with_route["route_earliest"]=="GP referral"],
                                            list_symptoms = list_of_symptoms_incl_prescriptions_by_location,
                                            number_cancers = len(list_ids_gp_referral),
                                            target_col = f"cancer_in_next_{days_ppv}_days")

df_ppv_sensitivity_with_dataset_gp["dataset"] = df_ppv_sensitivity_with_dataset_gp["S1_S2"].apply(lambda col_name : identify_val_in_list(col_name,list_datasets, null_return = "overall"))

df_ppv_sensitivity_with_dataset_gp

# COMMAND ----------

min_ppv_for_scatter = 1
min_sensitivity_for_scatter = 2

sns.set(font_scale = 1, style="whitegrid")

df = df_ppv_sensitivity_with_dataset_gp.copy()

df=df.sort_values(by="dataset")

# small number suppression
df["ppv (%)"] = np.where(df["total"]<=100, np.NaN, df["ppv (%)"])
df["sensitivity"] = np.where(df["count"]<=10, np.NaN, df["sensitivity"])

heatmap_data = df.pivot(index="S2", columns="S1", values="ppv (%)")

plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title("PPV (%) Heatmap for Symptom Pairs")
plt.show()


heatmap_data = df.pivot(index="S2", columns="S1", values="sensitivity")
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title("Sensitivity Heatmap for Symptom Pairs")
plt.show()

# COMMAND ----------


# scatterplot sensitivity vs ppv

df = df_ppv_sensitivity_with_dataset_gp.copy()

sns.set(font_scale = 1.2, style="white")
plt.figure(figsize=(10, 10))

df = df[df["ppv (%)"]>=min_ppv_for_scatter]
df=df[df["sensitivity"]>=min_sensitivity_for_scatter]
#df = df[df["S1"]!=df["S2"]]

sns.scatterplot(x="ppv (%)", y ="sensitivity", data = df, hue = "S1_S2", style = "S1_S2",markers = True, s=100)
plt.xlabel("PPV (%)")
plt.ylabel("Sensitivity (%)")

plt.legend(bbox_to_anchor = [1.05,1], loc = 2)
plt.xlim([0,4]);
plt.ylim([0, 50]);

plt.axhline(y=min_sensitivity_for_scatter, linestyle="--" )
plt.axvline(x=min_ppv_for_scatter, linestyle="--")

# scatterplot ppv vs avg_number_of_days_for_two_symptoms

sns.set(font_scale = 1.2, style="white")
plt.figure(figsize=(10, 10))

sns.scatterplot(x="ppv (%)", y ="avg_number_of_days_for_two_symptoms", data = df, hue = "S1_S2", style = "S1_S2",markers = True, s=100)
plt.xlabel("PPV (%)")
plt.ylabel("Average number of days between symptoms and diagnosis")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)
plt.ylim([0,365])

# COMMAND ----------

# MAGIC %md
# MAGIC # NG12 criteria for Chest X-ray and PPV

# COMMAND ----------

# MAGIC %md
# MAGIC ## time between latest symptom and diagnosis

# COMMAND ----------

# only consider unexplained symptoms for smokers
df_symptom_earliest_latest = df_flags_after_earliest_cancer_date[(df_flags_after_earliest_cancer_date["symptom_name"].isin(list_of_symptoms_non_critical + list_of_symptoms_critical)) & 
                                                                 (df_flags_after_earliest_cancer_date["Smoking_Flag"]==True)].groupby("Patient_ID")["date"].agg([np.min, np.max]).reset_index()

df_symptom_earliest_latest.columns = ["Patient_ID", "symptom_earliest", "symptom_latest"]


df_chest_xray_earliest_latest = df_all_flags_pd[(df_all_flags_pd["symptom_name"]=="chest_xray") & 
                                                (df_all_flags_pd["Smoking_Flag"]==True)].groupby("Patient_ID")["date"].agg([np.min, np.max]).reset_index()

df_chest_xray_earliest_latest.columns = ["Patient_ID", "xray_earliest", "xray_latest"]

df_compare_symptom_diagnosis = df_symptom_earliest_latest.merge(df_chest_xray_earliest_latest, on = "Patient_ID", how="left")

df_compare_symptom_diagnosis["days_between_symptom_earliest_xray_earliest"] = (df_compare_symptom_diagnosis['xray_earliest'] - df_compare_symptom_diagnosis['symptom_earliest']).dt.days
df_compare_symptom_diagnosis["days_between_symptom_earliest_xray_latest"] = (df_compare_symptom_diagnosis['xray_latest'] - df_compare_symptom_diagnosis['symptom_earliest']).dt.days
df_compare_symptom_diagnosis["days_between_symptom_latest_xray_earliest"] = (df_compare_symptom_diagnosis['xray_earliest'] - df_compare_symptom_diagnosis['symptom_latest']).dt.days
df_compare_symptom_diagnosis["days_between_symptom_latest_xray_latest"] = (df_compare_symptom_diagnosis['xray_latest'] - df_compare_symptom_diagnosis['symptom_latest']).dt.days

for col in ["days_between_symptom_earliest_xray_earliest", "days_between_symptom_earliest_xray_latest", "days_between_symptom_latest_xray_earliest", "days_between_symptom_latest_xray_latest"]:
    df_compare_symptom_diagnosis["abs_" + col] = df_compare_symptom_diagnosis[col].apply(lambda x:np.abs(x))


df_compare_symptom_diagnosis["fewest_days_abs"] = df_compare_symptom_diagnosis[["abs_days_between_symptom_earliest_xray_earliest", "abs_days_between_symptom_earliest_xray_latest","abs_days_between_symptom_latest_xray_earliest","abs_days_between_symptom_latest_xray_latest"]].min(axis=1)

# only consider those where the x-ray was after the symptom, or no x-ray at all
df_compare_symptom_diagnosis = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["symptom_latest"]<df_compare_symptom_diagnosis["xray_earliest"]) 
                                                            | (df_compare_symptom_diagnosis["xray_earliest"].isnull())]

df_compare_symptom_diagnosis["had_xray_symptom_within_range"] = np.where((df_compare_symptom_diagnosis["abs_days_between_symptom_latest_xray_latest"]<=30) | 
                                                                         (df_compare_symptom_diagnosis["abs_days_between_symptom_latest_xray_earliest"]<=30), True, False)

df_compare_symptom_diagnosis = df_compare_symptom_diagnosis.merge(df_cancer_site_complete_data_no_ltc_flag_pd, on = "Patient_ID", how="left")
df_compare_symptom_diagnosis["diagnosis_date_earliest"] = pd.to_datetime(df_compare_symptom_diagnosis["diagnosis_date_earliest"])
df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"] = (df_compare_symptom_diagnosis['diagnosis_date_earliest'] - df_compare_symptom_diagnosis['symptom_earliest']).dt.days
df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"] = (df_compare_symptom_diagnosis['diagnosis_date_earliest'] - df_compare_symptom_diagnosis['symptom_latest']).dt.days
df_compare_symptom_diagnosis[["days_between_symptom_earliest_diagnosis", "days_between_symptom_latest_diagnosis"]].describe()

df = df_flags_after_earliest_cancer_date[(df_flags_after_earliest_cancer_date["symptom_name"].isin(list_of_symptoms_non_critical + list_of_symptoms_critical)) & 
                                                                 (df_flags_after_earliest_cancer_date["Smoking_Flag"]==True)].copy()

df_latest_symptom = df.loc[df.groupby('Patient_ID')['date'].idxmax()]

df_latest_symptom = (
    df.sort_values('date')
      .drop_duplicates(subset='Patient_ID', keep='last')
)

df_compare_symptom_diagnosis = df_compare_symptom_diagnosis.merge(df_latest_symptom[["Patient_ID", "symptom_name", "dataset", "date"]], on = ["Patient_ID"], suffixes = ("", "_symptom"), how = "left")

df_compare_symptom_diagnosis

# COMMAND ----------

df_compare_symptom_diagnosis[["symptom_earliest", "symptom_latest", "xray_earliest", "xray_latest",  "diagnosis_date_earliest"]].describe(datetime_is_numeric=True)

# COMMAND ----------

ids_who_met_condition_for_xray_and_had_one = df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==True]["Patient_ID"].unique()
ids_who_met_condition_for_xray_and_did_not_have_one = df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==False]["Patient_ID"].unique()

ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==True) & 
                                                                                             (df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]<=365)]["Patient_ID"].unique()

ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==False) & 
                                                                                             (df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]<=365)]["Patient_ID"].unique()

ids_who_met_condition_but_had_xray_at_different_time = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (~df_compare_symptom_diagnosis["xray_earliest"].isnull())]["Patient_ID"].unique()


ids_cancer_who_met_condition_but_had_xray_at_different_time = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (~df_compare_symptom_diagnosis["xray_earliest"].isnull())
                                                                                    & (df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]<=365)]["Patient_ID"].unique()

ids_cancer_who_met_condition_but_never_had_xray = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (df_compare_symptom_diagnosis["xray_earliest"].isnull())
                                                                                    & (df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]<=365)]["Patient_ID"].unique()     


ids_who_met_condition_but_never_had_xray = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (df_compare_symptom_diagnosis["xray_earliest"].isnull())]["Patient_ID"].unique()

ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (df_compare_symptom_diagnosis["xray_earliest"].isnull())
                                                                                    & (df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]<=365) & 
                                                                                    (df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]<=30)]["Patient_ID"].unique()     

ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (df_compare_symptom_diagnosis["xray_earliest"].isnull())
                                                                                    & (df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]<=365) & 
                                                                                    (df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]>30)]["Patient_ID"].unique()     


num_patients_who_had_future_cancer_diagnosis_and_xray = len(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom)
num_patients_without_future_cancer_diagnosis_and_xray = len(ids_who_met_condition_for_xray_and_had_one) - num_patients_who_had_future_cancer_diagnosis_and_xray

num_patients_who_had_future_cancer_diagnosis_and_no_xray = len(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom)
num_patients_without_future_cancer_diagnosis_and_xray = len(ids_who_met_condition_for_xray_and_did_not_have_one) - num_patients_who_had_future_cancer_diagnosis_and_no_xray

num_patients_had_and_xray_but_at_different_time = len(ids_who_met_condition_but_had_xray_at_different_time)
num_patients_who_had_future_cancer_diagnosis_and_xray_but_at_different_time = len(ids_cancer_who_met_condition_but_had_xray_at_different_time)       

num_patients_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days = len(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)       


print("all ids who met condition", df_compare_symptom_diagnosis.shape[0])
print("all cancer ids who met condition", df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"]<=365].shape[0])

print("ids_who_met_condition_for_xray_and_had_one", len(ids_who_met_condition_for_xray_and_had_one))
print("num_patients_who_had_future_cancer_diagnosis_and_xray",num_patients_who_had_future_cancer_diagnosis_and_xray)

print("ids_who_met_condition_for_xray_and_did_not_have_one", len(ids_who_met_condition_for_xray_and_did_not_have_one))
print("num_patients_who_had_future_cancer_diagnosis_and_no_xray",num_patients_who_had_future_cancer_diagnosis_and_no_xray)
print("num_patients_had_and_xray_but_at_different_time", num_patients_had_and_xray_but_at_different_time)
print("num_patients_who_had_future_cancer_diagnosis_and_xray_but_at_different_time", num_patients_who_had_future_cancer_diagnosis_and_xray_but_at_different_time)
print("ids_who_met_condition_but_never_had_xray", len(ids_who_met_condition_but_never_had_xray))
print("ids_cancer_who_met_condition_but_never_had_xray", len(ids_cancer_who_met_condition_but_never_had_xray))

print("PPV: xray timely and cancer diagnosis: ", 100*num_patients_who_had_future_cancer_diagnosis_and_xray/len(ids_who_met_condition_for_xray_and_had_one))
print("PPV: no timely xray cancer diagnosis: ", 100*num_patients_who_had_future_cancer_diagnosis_and_no_xray/len(ids_who_met_condition_for_xray_and_did_not_have_one))
print("PPV: xray later and cancer diagnosis: ", 100*num_patients_who_had_future_cancer_diagnosis_and_xray_but_at_different_time/num_patients_had_and_xray_but_at_different_time)
print("PPV: no xray at all and cancer diagnosis: ", 100*len(ids_cancer_who_met_condition_but_never_had_xray)/len((ids_who_met_condition_but_never_had_xray)))

# COMMAND ----------

print(len(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom))
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------

print(len(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom))
#num_patients_who_had_future_cancer_diagnosis_and_no_xray
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------

print(len(ids_cancer_who_met_condition_but_had_xray_at_different_time))
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_had_xray_at_different_time)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_had_xray_at_different_time)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_had_xray_at_different_time)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------

print(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray)].shape)
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray)]["symptom_name"].value_counts(normalize=True))


# COMMAND ----------

print(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)].shape)
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------

print(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days)].shape)
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## time between earliest symptom and diagnosis

# COMMAND ----------

df_symptom_earliest_latest = df_all_flags_pd[(df_all_flags_pd["symptom_name"].isin(list_of_symptoms_non_critical + list_of_symptoms_critical)) & 
                                             (df_all_flags_pd["Smoking_Flag"]==True)].groupby("Patient_ID")["date"].agg([np.min, np.max]).reset_index()

df_symptom_earliest_latest.columns = ["Patient_ID", "symptom_earliest", "symptom_latest"]

df_symptom_earliest_latest = df_symptom_earliest_latest[(df_symptom_earliest_latest["symptom_earliest"]>=earliest_cancer_date) & 
                                                        (df_symptom_earliest_latest["symptom_earliest"]<latest_activity_date)]

df_chest_xray_earliest_latest = df_all_flags_pd[(df_all_flags_pd["symptom_name"]=="chest_xray") & 
                                                (df_all_flags_pd["Smoking_Flag"]==True)].groupby("Patient_ID")["date"].agg([np.min, np.max]).reset_index()

df_chest_xray_earliest_latest.columns = ["Patient_ID", "xray_earliest", "xray_latest"]

df_compare_symptom_diagnosis = df_symptom_earliest_latest.merge(df_chest_xray_earliest_latest, on = "Patient_ID", how="left")

df_compare_symptom_diagnosis["days_between_symptom_earliest_xray_earliest"] = (df_compare_symptom_diagnosis['xray_earliest'] - df_compare_symptom_diagnosis['symptom_earliest']).dt.days
df_compare_symptom_diagnosis["days_between_symptom_earliest_xray_latest"] = (df_compare_symptom_diagnosis['xray_latest'] - df_compare_symptom_diagnosis['symptom_earliest']).dt.days
df_compare_symptom_diagnosis["days_between_symptom_latest_xray_earliest"] = (df_compare_symptom_diagnosis['xray_earliest'] - df_compare_symptom_diagnosis['symptom_latest']).dt.days
df_compare_symptom_diagnosis["days_between_symptom_latest_xray_latest"] = (df_compare_symptom_diagnosis['xray_latest'] - df_compare_symptom_diagnosis['symptom_latest']).dt.days

for col in ["days_between_symptom_earliest_xray_earliest", "days_between_symptom_earliest_xray_latest", "days_between_symptom_latest_xray_earliest", "days_between_symptom_latest_xray_latest"]:
    df_compare_symptom_diagnosis["abs_" + col] = df_compare_symptom_diagnosis[col].apply(lambda x:np.abs(x))


df_compare_symptom_diagnosis["fewest_days_abs"] = df_compare_symptom_diagnosis[["abs_days_between_symptom_earliest_xray_earliest", "abs_days_between_symptom_earliest_xray_latest","abs_days_between_symptom_latest_xray_earliest","abs_days_between_symptom_latest_xray_latest"]].min(axis=1)

# only consider those where the x-ray was after the symptom, or no x-ray at all
df_compare_symptom_diagnosis = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["symptom_earliest"]<df_compare_symptom_diagnosis["xray_earliest"]) 
                                                            | (df_compare_symptom_diagnosis["xray_earliest"].isnull())]

df_compare_symptom_diagnosis["had_xray_symptom_within_range"] = np.where((df_compare_symptom_diagnosis["abs_days_between_symptom_earliest_xray_latest"]<=30) | 
                                                                         (df_compare_symptom_diagnosis["abs_days_between_symptom_earliest_xray_earliest"]<=30), True, False)

df_compare_symptom_diagnosis = df_compare_symptom_diagnosis.merge(df_cancer_site_complete_data_no_ltc_flag_pd, on = "Patient_ID", how="left")
df_compare_symptom_diagnosis["diagnosis_date_earliest"] = pd.to_datetime(df_compare_symptom_diagnosis["diagnosis_date_earliest"])
df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"] = (df_compare_symptom_diagnosis['diagnosis_date_earliest'] - df_compare_symptom_diagnosis['symptom_earliest']).dt.days
df_compare_symptom_diagnosis["days_between_symptom_latest_diagnosis"] = (df_compare_symptom_diagnosis['diagnosis_date_earliest'] - df_compare_symptom_diagnosis['symptom_latest']).dt.days

df = df_flags_after_earliest_cancer_date[(df_flags_after_earliest_cancer_date["symptom_name"].isin(list_of_symptoms_non_critical + list_of_symptoms_critical)) & 
                                                                 (df_flags_after_earliest_cancer_date["Smoking_Flag"]==True)].copy()

df_latest_symptom = df.loc[df.groupby('Patient_ID')['date'].idxmax()]

df_latest_symptom = (
    df.sort_values('date')
      .drop_duplicates(subset='Patient_ID', keep='last')
)

df_compare_symptom_diagnosis = df_compare_symptom_diagnosis.merge(df_latest_symptom[["Patient_ID", "symptom_name", "dataset", "date"]], on = ["Patient_ID"], suffixes = ("", "_symptom"), how = "left")

df_compare_symptom_diagnosis

# COMMAND ----------

df_compare_symptom_diagnosis[["symptom_earliest", "symptom_latest", "xray_earliest", "xray_latest",  "diagnosis_date_earliest"]].describe(datetime_is_numeric=True)

# COMMAND ----------

df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==False) & 
                                                                                             (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=max_days_to_diagnosis_from_symptom)].shape

# COMMAND ----------

ids_who_met_condition_for_xray_and_had_one = df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==True]["Patient_ID"].unique()
ids_who_met_condition_for_xray_and_did_not_have_one = df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==False]["Patient_ID"].unique()

ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==True) & 
                                                                                             (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=max_days_to_diagnosis_from_symptom)]["Patient_ID"].unique()

ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["had_xray_symptom_within_range"]==False) & 
                                                                                             (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=max_days_to_diagnosis_from_symptom)]["Patient_ID"].unique()

ids_who_met_condition_but_had_xray_at_different_time = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (~df_compare_symptom_diagnosis["xray_earliest"].isnull())]["Patient_ID"].unique()


ids_cancer_who_met_condition_but_had_xray_at_different_time = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (~df_compare_symptom_diagnosis["xray_earliest"].isnull())
                                                                                    & (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=max_days_to_diagnosis_from_symptom)]["Patient_ID"].unique()

ids_cancer_who_met_condition_but_never_had_xray = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (df_compare_symptom_diagnosis["xray_earliest"].isnull())
                                                                                    & (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=max_days_to_diagnosis_from_symptom)]["Patient_ID"].unique()     


ids_who_met_condition_but_never_had_xray = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (df_compare_symptom_diagnosis["xray_earliest"].isnull())]["Patient_ID"].unique()

ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (df_compare_symptom_diagnosis["xray_earliest"].isnull())
                                                                                    & (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=max_days_to_diagnosis_from_symptom) & 
                                                                                    (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=30)]["Patient_ID"].unique()     

ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days = df_compare_symptom_diagnosis[(df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_met_condition_for_xray_and_did_not_have_one))
                                                                                    & (df_compare_symptom_diagnosis["xray_earliest"].isnull())
                                                                                    & (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=max_days_to_diagnosis_from_symptom) & 
                                                                                    (df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]>30)]["Patient_ID"].unique()     


num_patients_who_had_future_cancer_diagnosis_and_xray = len(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom)
num_patients_without_future_cancer_diagnosis_and_xray = len(ids_who_met_condition_for_xray_and_had_one) - num_patients_who_had_future_cancer_diagnosis_and_xray

num_patients_who_had_future_cancer_diagnosis_and_no_xray = len(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom)
num_patients_without_future_cancer_diagnosis_and_xray = len(ids_who_met_condition_for_xray_and_did_not_have_one) - num_patients_who_had_future_cancer_diagnosis_and_no_xray

num_patients_had_and_xray_but_at_different_time = len(ids_who_met_condition_but_had_xray_at_different_time)
num_patients_who_had_future_cancer_diagnosis_and_xray_but_at_different_time = len(ids_cancer_who_met_condition_but_had_xray_at_different_time)       

num_patients_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days = len(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)       


print("all ids who met condition", df_compare_symptom_diagnosis.shape[0])
print("all cancer ids who met condition", df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["days_between_symptom_earliest_diagnosis"]<=max_days_to_diagnosis_from_symptom].shape[0])

print("ids_who_met_condition_for_xray_and_had_one", len(ids_who_met_condition_for_xray_and_had_one))
print("num_patients_who_had_future_cancer_diagnosis_and_xray",num_patients_who_had_future_cancer_diagnosis_and_xray)

print("ids_who_met_condition_for_xray_and_did_not_have_one", len(ids_who_met_condition_for_xray_and_did_not_have_one))
print("num_patients_who_had_future_cancer_diagnosis_and_no_xray",num_patients_who_had_future_cancer_diagnosis_and_no_xray)
print("num_patients_had_and_xray_but_at_different_time", num_patients_had_and_xray_but_at_different_time)
print("num_patients_who_had_future_cancer_diagnosis_and_xray_but_at_different_time", num_patients_who_had_future_cancer_diagnosis_and_xray_but_at_different_time)
print("ids_who_met_condition_but_never_had_xray", len(ids_who_met_condition_but_never_had_xray))
print("ids_cancer_who_met_condition_but_never_had_xray", len(ids_cancer_who_met_condition_but_never_had_xray))

print("PPV: xray timely and cancer diagnosis: ", 100*num_patients_who_had_future_cancer_diagnosis_and_xray/len(ids_who_met_condition_for_xray_and_had_one))
print("PPV: no timely xray cancer diagnosis: ", 100*num_patients_who_had_future_cancer_diagnosis_and_no_xray/len(ids_who_met_condition_for_xray_and_did_not_have_one))
print("PPV: xray later and cancer diagnosis: ", 100*num_patients_who_had_future_cancer_diagnosis_and_xray_but_at_different_time/num_patients_had_and_xray_but_at_different_time)
print("PPV: no xray at all and cancer diagnosis: ", 100*len(ids_cancer_who_met_condition_but_never_had_xray)/len((ids_who_met_condition_but_never_had_xray)))

# COMMAND ----------

print(len(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom))
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_had_xray_and_cancer_diagnosis_in_1_year_after_symptom)]["symptom_name"].value_counts(normalize=True))


# COMMAND ----------

print(len(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom))
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_who_did_not_have_xray_and_had_cancer_diagnosis_in_1_year_after_symptom)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------

print(len(ids_cancer_who_met_condition_but_had_xray_at_different_time))
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_had_xray_at_different_time)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_had_xray_at_different_time)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_had_xray_at_different_time)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------

print(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray)].shape)
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray)]["symptom_name"].value_counts(normalize=True))


# COMMAND ----------

print(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)].shape)
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_within_30_days)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------

print(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days)].shape)
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days)]["tumour_stage_earliest_first_char"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days)]["route_earliest"].value_counts(normalize=True).sort_values())
print("\n")
display(df_compare_symptom_diagnosis[df_compare_symptom_diagnosis["Patient_ID"].isin(ids_cancer_who_met_condition_but_never_had_xray_but_diagnosed_after_30_days)]["symptom_name"].value_counts(normalize=True))

# COMMAND ----------


