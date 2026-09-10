# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Lung Cancer Patient Event Construction
# MAGIC
# MAGIC The purpose of this notebook is to link together several different datasets to create a complete medical record for lung cancer patients in the period before their diagnosis. Data sources include primary care, secondary care, medication prescriptions, emergency attendances, and demographic factors. 

# COMMAND ----------

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

# COMMAND ----------

cancer_site = "Lung"
rerun_cohort_tables = True
write_output = True

version = config_pathways.cancer_site_mappings[cancer_site]["run_version"]

earliest_cancer_date = "2022-05-01"
latest_cancer_date = "2025-01-01"

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

# COMMAND ----------

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


# COMMAND ----------

# get the date of the earliest recorded cancer diagnosis, per patient
df_id_first_cancer = df_cancer_by_group.groupBy("PSEUDO_NHS_NUMBER").agg(F.min("diagnosis_date_earliest").alias("diagnosis_date_earliest"))


# COMMAND ----------

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

# COMMAND ----------

# get the first character of the tumour stage (1,2,3 or 4). This removes the information such as stage 2A etc
df_cancer_site = df_cancer_site.withColumn("tumour_stage_earliest_first_char",
                                           F.substring("tumour_stage_earliest", 1, 1))

df_cancer_site = df_cancer_site.withColumnRenamed("PSEUDO_NHS_NUMBER", "Patient_ID")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cancers over time (data visualisation)

# COMMAND ----------

# plot number of breast cancer diagnoses over time
df_cancers_by_year_month = df_cancer_site.groupBy("YearMonth").count().orderBy("YearMonth").toPandas()
df_cancers_by_stage_year_month = df_cancer_site.groupBy(["YearMonth", "tumour_stage_earliest_first_char"]).count().orderBy("YearMonth").toPandas()

# COMMAND ----------

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.4)
sns.set_style("white")
sns.lineplot(
    x="YearMonth", y="count",data=df_cancers_by_year_month
)
plt.xticks(rotation = 90);

plt.ylim(bottom=0)

# COMMAND ----------

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.4)
sns.set_style("white")
sns.lineplot(
    x="YearMonth", y="count",hue="tumour_stage_earliest_first_char",data=df_cancers_by_stage_year_month,
    hue_order = ["1","2","3","4"]
)
plt.xticks(rotation = 90);
plt.ylim(bottom=0);
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

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

if rerun_cohort_tables == True:

    dict_cohort_paths = processing.get_latest_dict_cohort_paths(containerName = config.containerName_bronze,
                                                                lakeName = config.lakeName,
                                                                filePath = "",
                                                                cohort_name = "pcp_Cohort_")
                                                                
    first_date = "2021-03-30"

    i = 0 
    list_all_cohort_dates = list(dict_cohort_paths.keys())

    for cohort_date, path in dict_cohort_paths.items():

        print("Processing cohort ", path)

        #import LDM snapshot for "cohort_date" 
        df_new_cohort = read_parquet_file(containerName=config.containerName_bronze,
                                        lakeName=config.lakeName,
                                        filePath=path)
        
        df_new_cohort = processing.identify_and_remove_duplicate_ids(df_new_cohort, cohort_date)
            
        # Creating a patient level dataframe
        df_cohort = df_new_cohort\
            .select("Patient_ID", "Cohort_Date", "Age")\
            .withColumnRenamed("Cohort_Date", "Cohort_Date_"+ cohort_date)\
            .withColumnRenamed("Age", "Age_"+ cohort_date)

        if i == 0: # first time picking up a cohort

            # initiate df_all_cohorts    
            df_all_cohorts = df_cohort          

            # initiate df_multi_cohorts
            if cohort_date >= first_date:
                print("Adding cohort ", cohort_date, " to multi cohort table")  
                df_multi_cohorts = df_new_cohort
            else: 
                print("Not adding cohort ", cohort_date, " to multi cohort table as it is outside of the first date to last date range")
                schema = df_new_cohort.schema
                df_multi_cohorts = spark.createDataFrame([], schema=StructType(schema))

        else: # subsequent cohorts
            
            #join the subsequent cohort to the list with all cohorts
            df_all_cohorts = df_all_cohorts.join(df_cohort, on = "Patient_ID", how="fullouter")
            
            # join the subsequent cohort to the list with all cohorts
            if cohort_date >= first_date:
                print("Adding cohort ", cohort_date, " to multi cohort table")                 
                df_multi_cohorts = df_multi_cohorts.unionByName(df_new_cohort, allowMissingColumns=True)

            else:
                print("Not adding cohort ", cohort_date, " to multi cohort table as it is outside of the first date to last date range")

        print("All cohorts size:", df_all_cohorts.count())
        print("All cohorts distinct patients:",df_all_cohorts.select("Patient_ID").distinct().count())
        print("Multi cohorts size:", df_multi_cohorts.count())
        print("Multi cohorts distinct patients:", df_multi_cohorts.select("Patient_ID").distinct().count())
        print("-------------------------------------------------------------")

        i += 1

else:
    df_multi_cohorts = read_parquet_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= f""
    
    df_all_cohorts = read_parquet_file(containerName =config.containerName_platinum, 
                                   lakeName=config.lakeName,
                                   filePath= f""

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

# COMMAND ----------

if "Smoking_Flag" in df_multi_cohorts.columns:
    df_multi_cohorts = df_multi_cohorts.drop("Smoking_Flag")
    
df_multi_cohorts = df_multi_cohorts.join(df_smoking_flags, on="Patient_ID", how="left")
df_multi_cohorts = df_multi_cohorts.fillna(False, subset="Smoking_Flag")

# COMMAND ----------

display(df_all_cohorts)

# COMMAND ----------

df_all_cohorts.count()

# COMMAND ----------

# only keep rows for patients with a cancer diagnosis
df_all_cohorts_selected_ids = df_all_cohorts.join(df_cancer_site.select(["Patient_ID", "diagnosis_date_earliest"]),
                                                  on = "Patient_ID",
                                                  how="inner")

# COMMAND ----------

df_all_cohorts_selected_ids = df_all_cohorts_selected_ids.withColumn(
    "history_start",
    F.date_add(F.col("diagnosis_date_earliest"),
    -365)
)

# COMMAND ----------

display(df_all_cohorts_selected_ids)

# COMMAND ----------

# add the diagnosis date information to the multi cohort table
df_multi_cohorts_selected_ids = df_multi_cohorts.join(df_cancer_site.select(["Patient_ID", "diagnosis_date_earliest"]),
                                                      on = "Patient_ID",
                                                      how="inner")

# COMMAND ----------

df_multi_cohorts_selected_ids = df_multi_cohorts_selected_ids.withColumn(
    "days_between_diagnosis_and_cohort",
    F.datediff(
        F.col("diagnosis_date_earliest"),
        F.col("Cohort_Date")
    )
)

# COMMAND ----------

df_multi_cohorts_selected_ids = df_multi_cohorts_selected_ids.withColumn(
    "history_start",
    F.date_sub(F.col("diagnosis_date_earliest"), 365)
)

# COMMAND ----------

display(df_multi_cohorts_selected_ids)

# COMMAND ----------

# only keep cohort data in the year prior to the diagnosis
df_multi_cohorts_selected_ids_in_last_year = df_multi_cohorts_selected_ids.filter(
    (F.col("days_between_diagnosis_and_cohort")>=0) & (F.col("days_between_diagnosis_and_cohort")<365))

# COMMAND ----------

# count how many cohorts the patient was included in the last 1 year
df_count_cohorts_in_last_year = df_multi_cohorts_selected_ids_in_last_year.groupby("Patient_ID").count()

# COMMAND ----------

df_four_quarters_in_last_year = df_count_cohorts_in_last_year.filter(F.col("count")>=4)

# COMMAND ----------

print("Percentage who were in cohort table in last 4 quarters: ", 100*(df_four_quarters_in_last_year.count())/(df_count_cohorts_in_last_year.count()))

# COMMAND ----------

# updated cancer site table -> only those with at least 4 quarters in last year

df_cancer_site = df_cancer_site.join(df_four_quarters_in_last_year.select("Patient_ID"), on = "Patient_ID", how="inner")


# COMMAND ----------

df_cancer_site.count()

# COMMAND ----------

# identify and keep the latest cohort 
window_spec = Window.partitionBy("Patient_ID").orderBy(F.col("days_between_diagnosis_and_cohort").asc())
df_latest_cohort = df_multi_cohorts_selected_ids_in_last_year.withColumn("row_num", F.row_number().over(window_spec)).filter(F.col("row_num") == 1).drop("row_num")

# remove any patients with a LTC_Cancer flag
df_latest_cohort = df_latest_cohort.filter(F.col("LTC_Cancer")==0)

df_latest_cohort = df_latest_cohort.join(df_cancer_site.select("Patient_ID"), on = "Patient_ID", how="inner")
display(df_latest_cohort)

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

display(df_cancer_site)

# COMMAND ----------

df_cancer_site.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## GP events

# COMMAND ----------

df_sct_concept_definitions = read_parquet_file(containerName =config.containerName_bronze, 
                                               lakeName= config.lakeName,
                                               filePath= "") 

# COMMAND ----------

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

# COMMAND ----------

# only keeping GP events for cancer patients
df_gp_events = df_gp_events.join(df_cancer_site.select("Patient_ID"), on = "Patient_ID", how="inner")

df_gp_events = df_gp_events.withColumn("dataset", F.lit("gp_events"))
df_gp_events = df_gp_events.withColumn("date", F.col("Attendance_Date"))

# COMMAND ----------

df_gp_events = df_gp_events.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                 df_gp_events.SnomedCode == df_chosen_snomed_description.Concept_ID,
                                 how="left" )

df_gp_events = df_gp_events.filter(F.col("SnomedCode") != "-1") # remove where code is -1

# COMMAND ----------

# remove duplicates
df_gp_events = df_gp_events.dropDuplicates(["Patient_ID", "date", "SnomedCode"])

# COMMAND ----------

df_gp_events = df_gp_events.withColumn("description", F.col("Term"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SUS

# COMMAND ----------

df_sus_all = read_parquet_file(containerName =config.containerName_bronze, 
                             lakeName=config.lakeName,
                             filePath= config.filePath_sus)

# COMMAND ----------

df_sus_all = df_sus_all.join(df_cancer_site.select("Patient_ID"),on = "Patient_ID", how="inner")
df_sus_all = df_sus_all.withColumn("dataset", F.lit("SUS_Activity_Extract"))
df_sus_all = df_sus_all.withColumn("date", F.col("Attendance_Date"))

# COMMAND ----------

all_columns = df_sus_all.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wide to long transformation (each diagnosis, procedure, HRG on new row)

# COMMAND ----------

# Define diagnosis/procedure column pairs
diagnosis_cols = ['Primary_Diagnosis'] + [(f"Secondary_Diagnosis_{i}") for i in range(1, 13)]
procedure_cols = ['Primary_Procedure_Code'] + [(f"Secondary_Procedure_Code_{i}") for i in range(1, 13)]
hrg_cols = ['HRG_Code']

# columns to keep after wide to long transformation (all columns minus the ones going to rows)
demographic_cols = list(set(all_columns)- set(diagnosis_cols)- set(procedure_cols)- set(hrg_cols))

# COMMAND ----------

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

# COMMAND ----------

df_sus_all_events = df_sus_all_events.withColumn("Event_Code", F.regexp_replace(F.col("Event_Code"), r"\.", ""))

# COMMAND ----------

# remove duplicates 
df_sus_all_events = df_sus_all_events.dropDuplicates(["Patient_ID", "date", "Event_Code", "Event_Type"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Link to ICD10 mapping

# COMMAND ----------

df_icd10_ref = read_parquet_file(containerName =config.containerName_bronze, 
                             lakeName= config.lakeName,
                             filePath= config.filePath_icd10) 

df_icd10_ref_cat_3 = processing.process_icd10_ref(df_icd10_ref)

# COMMAND ----------

df_sus_all_events = df_sus_all_events.withColumn(
                                    "Diagnosis_type",
                                   F.when(
                                    F.col("Event_Type") == "Diagnosis_SUS",
                                    F.when(F.regexp_extract(F.col("Event_Code"), r'^[A-Z][0-9][0-9A-Z]?$|^[A-Z][0-9][0-9A-Z]+$', 0) != "", "ICD10")
                                   .when(F.regexp_extract(F.col("Event_Code"), r'^[0-9]{6,18}$', 0) != "", "SNOMED")
                                   .otherwise("Other")
                                   ).otherwise("")
)

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Link to Procedure (OPCS)

# COMMAND ----------

df_opcs_ref = read_parquet_file(containerName =config.containerName_bronze, 
                             lakeName= config.lakeName,
                             filePath= "") 


# COMMAND ----------

w_latest = Window.partitionBy("Code_Without_Decimal").orderBy(F.desc("Effective_to"))

df_opcs_ref_latest = (
    df_opcs_ref
    .withColumn("row_num", F.row_number().over(w_latest))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
)


# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Link to SNOMED

# COMMAND ----------

df_sus_all_events = df_sus_all_events.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                           (df_sus_all_events.Event_Code == df_chosen_snomed_description.Concept_ID) & ((df_sus_all_events.Diagnosis_type == "SNOMED") | ((df_sus_all_events.Procedure_type == "SNOMED"))),
                                           "left")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Link to HRG Code

# COMMAND ----------

df_hrg = read_parquet_file(containerName =config.containerName_bronze, 
                             lakeName= config.lakeName,
                             filePath= "") 

df_hrg_latest = df_hrg.filter(F.col("Is_Latest")==1)

# COMMAND ----------

df_sus_all_events = df_sus_all_events.join(
    df_hrg_latest.select(["HRG_Code", "HRG_Name"]),
    (df_sus_all_events.Event_Code == df_hrg_latest.HRG_Code) &
    (df_sus_all_events.Event_Type=="HRG_SUS"),
    "left"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Create description column

# COMMAND ----------

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

# COMMAND ----------

display(df_sus_all_events)

# COMMAND ----------

print(df_sus_all_events.count())

# COMMAND ----------

df_sus_all_events.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Acute

# COMMAND ----------

df_acute = read_parquet_file(containerName =config.containerName_bronze, 
                             lakeName=config.lakeName,
                             filePath=config.filePath_acute)

# COMMAND ----------

df_acute = df_acute.join(df_cancer_site.select("Patient_ID"),on = "Patient_ID", how="inner")
df_acute = df_acute.withColumn("dataset", F.lit("Acute_All"))
df_acute = df_acute.withColumn("date", F.col("Attendance_Date"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wide to long transformation (each diagnosis, procedure, HRG on new row)

# COMMAND ----------

all_columns = df_acute.columns

# Define diagnosis/procedure column pairs
hrg_cols = ["Dimention_5"]
diagnosis_cols = ["Dimention_6"]
procedure_cols = ["Dimention_7"]


# columns to keep after wide to long transformation (all columns minus the ones going to rows)
demographic_cols = list(set(all_columns)- set(hrg_cols) - set(diagnosis_cols) - set(procedure_cols))

# COMMAND ----------

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

# COMMAND ----------

df_acute_all_events = df_acute_all_events.withColumn("Event_Code", F.regexp_replace(F.col("Event_Code"), r"\.", ""))

# COMMAND ----------

df_acute_all_events = df_acute_all_events.dropDuplicates(["Patient_ID", "date", "Event_Code", "Event_Type"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Link to ICD10 mapping

# COMMAND ----------

df_acute_all_events = df_acute_all_events.withColumn(
                                    "Diagnosis_type",
                                   F.when(
                                    F.col("Event_Type") == "Diagnosis_ACUTE",
                                    F.when(F.regexp_extract(F.col("Event_Code"), r'^[A-Z][0-9][0-9A-Z]?$|^[A-Z][0-9][0-9A-Z]+$', 0) != "", "ICD10")
                                   .when(F.regexp_extract(F.col("Event_Code"), r'^[0-9]{6,18}$', 0) != "", "SNOMED")
                                   .otherwise("Other")
                                   ).otherwise("")
)

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Link to Procedure (OPCS)

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Link to SNOMED

# COMMAND ----------

df_acute_all_events = df_acute_all_events.join(df_chosen_snomed_description.select(["Concept_ID", "Term"]),
                                           (df_acute_all_events.Event_Code == df_chosen_snomed_description.Concept_ID) & ((df_acute_all_events.Diagnosis_type == "SNOMED") | ((df_acute_all_events.Procedure_type == "SNOMED"))),
                                           "left")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Link to HRG Code

# COMMAND ----------

df_acute_all_events = df_acute_all_events.join(
    df_hrg_latest.select(["HRG_Code", "HRG_Name"]),
    (df_acute_all_events.Event_Code == df_hrg_latest.HRG_Code) &
    (df_acute_all_events.Event_Type=="HRG_ACUTE"),
    "left"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create description

# COMMAND ----------

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

# COMMAND ----------

print(df_acute_all_events.count())

# COMMAND ----------

df_acute_all_events.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Union SUS and Acute, and remove duplicates

# COMMAND ----------

df_sus_acute_all_events = df_acute_all_events.unionByName(df_sus_all_events, allowMissingColumns=True)



# COMMAND ----------

df_sus_acute_all_events = df_sus_acute_all_events.dropDuplicates(["Patient_ID","date","Diagnosis_type", "Procedure_type", "HRG_Name", "Event_Code"])

# COMMAND ----------

print(df_sus_acute_all_events.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## OoH call

# COMMAND ----------

df_OoH = read_parquet_file(containerName =config.containerName_bronze, 
                           lakeName=config.lakeName,
                           filePath=config.filePath_OoH)

# COMMAND ----------

df_OoH = df_OoH.join(df_cancer_site.select("Patient_ID"), on = "Patient_ID", how="inner")
df_OoH = df_OoH.withColumn("dataset", F.lit("UC_OoH_All"))
df_OoH = df_OoH.withColumn("date", F.col("Attendance_Date"))

# COMMAND ----------

# remove duplicates
df_OoH = df_OoH.distinct()

# COMMAND ----------

display(df_OoH)

# COMMAND ----------

df_OoH = df_OoH.withColumn("description", 
                           F.concat_ws("",F.lit("Contact Type: "), F.col("Dimention_1"), F.lit(" | ") , F.lit("Outcome: "), F.col("Dimention_2"))
) 


# COMMAND ----------

df_OoH.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC_999
# MAGIC

# COMMAND ----------

df_999 = read_parquet_file(containerName =config.containerName_bronze, 
                           lakeName=config.lakeName,
                           filePath=config.filePath_999)

# COMMAND ----------

df_999 = df_999.join(df_cancer_site.select("Patient_ID"),on = "Patient_ID", how="inner")
df_999 = df_999.withColumn("dataset", F.lit("UC_999_All"))
df_999 = df_999.withColumn("date", F.col("Attendance_Date"))

# COMMAND ----------

df_999 = df_999.withColumn("description", 
                           F.concat_ws("",F.lit("Chief Complaint: "), F.col("Dimention_1"), F.lit(" | ") , F.lit("Response: "), F.col("Dimention_2"))
) 


# COMMAND ----------

# drop duplicates
df_999 = df_999.dropDuplicates(["Patient_ID","date","Dimention_1", "Dimention_2"])

# COMMAND ----------

display(df_999)

# COMMAND ----------

df_999.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ##111

# COMMAND ----------

df_111 = read_parquet_file(containerName =config.containerName_bronze, 
                           lakeName=config.lakeName,
                           filePath=config.filePath_111)

# COMMAND ----------

df_111_reference = read_csv_file(containerName =config.containerName_platinum, 
                                     lakeName= config.lakeName,
                                     filePath= config.filePath_111_mapping)

# COMMAND ----------

df_111 = df_111.join(df_cancer_site.select("Patient_ID"),on = "Patient_ID", how="inner")
df_111 = df_111.withColumn("dataset", F.lit("UC_111_All"))
df_111 = df_111.withColumn("date", F.col("Attendance_Date"))

# COMMAND ----------

df_111_data = processing.process_111_dataset(df_111, df_111_reference)

# COMMAND ----------

df_111_data = df_111_data.withColumn("description", 
                           F.concat_ws("",F.lit("Symptom: "), F.col("SG_Description"),
                                    F.lit(" | ") , F.lit("SD_Description: "), F.col("SD_Description"),
                                    F.lit(" | ") , F.lit("DX_Description: "), F.col("DX_Description"))
) 


# COMMAND ----------

df_111_data = df_111_data.withColumn("SG_Description", 
                                    F.when(F.col("SG_Description").isNull(), "Unknown")
                                    .otherwise(F.col("SG_Description"))
                                    )

# COMMAND ----------

df_111_data = df_111_data.dropDuplicates(["Patient_ID","date","SG_Description", "SD_Description", "DX_Description"])

# COMMAND ----------

display(df_111_data)

# COMMAND ----------

df_111_data.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## ECDS data

# COMMAND ----------

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

# COMMAND ----------

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

# COMMAND ----------

display(df_ecds_data_chief_complaint)

# COMMAND ----------

df_ecds_data_chief_complaint = df_ecds_data_chief_complaint.withColumn("description", 
                                                                       F.concat_ws("",F.lit("Chief_Complaint: "), F.col("Emergency_Care_Chief_Complaint"),
                                                                                   F.lit(" | ") , F.lit("Complaint Extended Term: "), F.col("Emergency_Care_Chief_Complaint_Extended_Term"),
                                                                                   F.lit(" | ") , F.lit("Acuity: "), F.col("Emergency_Care_Acuity"),
                                                                                   F.lit(" | ") , F.lit("Primary Diagnosis: "), F.col("Primary_Diagnosis_ECDS"),
                                                                                   F.lit(" | ") , F.lit("Procedure: "), F.col("Primary_Procedure_Code_ECDS"),
                                                                                   F.lit(" | ") , F.lit("Discharge_Status: "), F.col("Emergency_Care_Discharge_Status"),
                                                                                   F.lit(" | ") , F.lit("Discharge_Follow_Up: "), F.col("Emergency_Care_Discharge_Follow_Up"))
                                                                       )





# COMMAND ----------

display(df_ecds_data_chief_complaint.groupby("Emergency_Care_Chief_Complaint").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## GP appointments

# COMMAND ----------

df_gp_appointments = read_parquet_file(containerName =config.containerName_bronze, 
                                       lakeName=config.lakeName,
                                       filePath=config.filePath_gp_all)

df_gp_appointments = df_gp_appointments.join(df_cancer_site.select("Patient_ID"),on = "Patient_ID", how="inner")
df_gp_appointments = df_gp_appointments.withColumn("dataset", F.lit("GP_Appointments"))
df_gp_appointments = df_gp_appointments.withColumn("date", F.col("Attendance_Date"))
df_gp_appointments = df_gp_appointments.withColumn("description", F.col("Record_Classification"))

# COMMAND ----------

display(df_gp_appointments)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GP Meds

# COMMAND ----------

df_gp_meds = read_parquet_file(containerName =config.containerName_bronze, 
                                       lakeName=config.lakeName,
                                       filePath=config.filePath_gp_meds)

# COMMAND ----------

# only keeping GP meds for cancer patients
df_gp_meds = df_gp_meds.join(df_cancer_site.select("Patient_ID"), on = "Patient_ID", how="inner")

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

# COMMAND ----------

df_gp_meds = df_gp_meds.withColumn("description", F.col("Term"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Union tables

# COMMAND ----------

df_all_activity = df_gp_events.unionByName(df_sus_acute_all_events,allowMissingColumns=True)\
                            .unionByName(df_OoH, allowMissingColumns=True)\
                            .unionByName(df_999.drop("Decision_to_Refer_to_Service_Date", "Discharge_Date"), allowMissingColumns=True)\
                            .unionByName(df_111_data.drop("Decision_to_Refer_to_Service_Date", "Discharge_Date"), allowMissingColumns=True)\
                            .unionByName(df_cancer_site.drop("diagnosis_date_earliest"), allowMissingColumns=True)\
                            .unionByName(df_ecds_data_chief_complaint, allowMissingColumns=True)\
                            .unionByName(df_gp_meds, allowMissingColumns=True)\
                            .unionByName(df_gp_appointments.drop("Decision_to_Refer_to_Service_Date", "Discharge_Date"), allowMissingColumns=True)


# COMMAND ----------

# add the cancer diagnosis date
df_all_activity = df_all_activity.join(df_cancer_site.select(["Patient_ID", "diagnosis_date_earliest"]), on = "Patient_ID", how="inner")
df_all_activity = df_all_activity.withColumn("days_between_activity_diagnosis", F.datediff(F.col("diagnosis_date_earliest"), F.col("date")))


# COMMAND ----------

# only keep activity from patients included in analysis
df_all_activity = df_all_activity.join(df_latest_cohort.select(["Patient_ID"]), on = "Patient_ID", how="inner")


# COMMAND ----------

display(df_all_activity.groupby("dataset").count())

# COMMAND ----------

# remove duplicate rows
df_all_activity = df_all_activity.distinct()

# COMMAND ----------

df_all_activity.cache()
print(df_all_activity.count())

# COMMAND ----------

display(df_all_activity.groupby("dataset").count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Data exports

# COMMAND ----------

if write_output == True:

    fullPath_output=""

    df_latest_cohort.write.mode("overwrite").parquet(
        fullPath_output,
    )

    fullPath_output=""

    df_cancer_site.write.mode("overwrite").parquet(
        fullPath_output,
    )

    fullPath_output=""

    df_all_activity.write.mode("overwrite").parquet(
        fullPath_output,
    )



# COMMAND ----------

if rerun_cohort_tables == True:
    fullPath_output=""

    df_multi_cohorts.write.mode("overwrite").parquet(
        fullPath_output,
    )

    fullPath_output=""

    df_all_cohorts.write.mode("overwrite").parquet(
        fullPath_output,
    )
