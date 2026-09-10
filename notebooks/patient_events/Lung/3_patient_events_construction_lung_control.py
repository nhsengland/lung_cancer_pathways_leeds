# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Construct controls for lung cancer cases
# MAGIC
# MAGIC The notebook constructs a patient record across different services for a control cohort chosen to demographically match the lung cancer patients. The process for constructing this cohort is as follows.
# MAGIC - For each cancer case, identify a patient of the same age band, sex and GP practice at the same point in time as the cancer diagnosis
# MAGIC - For cases that do not find a match base on those criteria, drop the requirement for the control to be from the same GP practice
# MAGIC - Export the same datasets (patient level and activity datasets) as the counterpart patient_events_construction notebook for the lung cancer cases

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
control_site = "Lung_control"
rerun_selection_controls = True
write_output = True

version = config_pathways.cancer_site_mappings[cancer_site]["run_version"]

earliest_cancer_date = "2022-05-01"
latest_cancer_date = "2025-01-01"

# COMMAND ----------

df_patient_flags = read_parquet_file(containerName =config.containerName_platinum, 
                                     lakeName=config.lakeName,
                                     filePath= f"")

display(df_patient_flags)

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
                                filePath= f""

df_all_cohorts = read_parquet_file(containerName =config.containerName_platinum, 
                                lakeName=config.lakeName,
                                filePath= f"")

# COMMAND ----------

# run to select new matched controls
if rerun_selection_controls == True:

    only_ids_with_full_history = df_all_cohorts.dropna().select("Patient_ID")
    only_ids_with_full_history = only_ids_with_full_history.distinct()

    # only keep those with full history: 
    df_multi_cohorts = df_multi_cohorts.join(only_ids_with_full_history, on = "Patient_ID", how="inner")

    df_multi_cohorts = df_multi_cohorts.withColumn(
        "imd_decile_group",
        F.when(F.col("IMD_Decile").isin("1", "2", "3"), "decile_1_to_3")
        .when(F.col("IMD_Decile").isin("4", "5", "6", "7"), "decile_4_to_7")
        .when(F.col("IMD_Decile").isin("8", "9", "10"), "decile_8_to_10")
        .otherwise("unknown")
    )

    df_multi_cohorts_exclude_existing = df_multi_cohorts.join(df_patient_flags, on = "Patient_ID", how="leftanti")

    df_patient_flags = df_patient_flags.withColumnRenamed("Patient_ID", "Patient_ID_" + cancer_site)

    # pick one ID, per gender, per age, per practice
    joining_fields = ["Cohort_Date", "Smoking_Flag", "Practice_Code", "AgeBand_5yr", "Sex"]

    df_potential_controls = df_multi_cohorts_exclude_existing.join(df_patient_flags.select(["Patient_ID_"+cancer_site, "diagnosis_date_earliest"] + joining_fields),
                                                                   on = joining_fields,
                                                                   how="inner")

    # One control to match with one case, only once in time (i.e. prevent the same control matching with a difference case at a different point in time)  
    df_potential_controls = df_potential_controls.dropDuplicates(subset = ["Patient_ID"])

    print("Potential controls: ", df_potential_controls.count())
    print("Potential controls unique IDs: ", df_potential_controls.select("Patient_ID").distinct().count())
    print("Cases with control match: ", df_potential_controls.select("Patient_ID_"+cancer_site).distinct().count())

    # pick the ID
    window = Window.partitionBy("Patient_ID_" + cancer_site).orderBy(F.col("Patient_ID"))

    df_selected_control = df_potential_controls.withColumn("row",F.row_number().over(window)) \
    .filter(F.col("row") == 1).drop("row")

    df_selected_control.cache()
    print("Number of selected controls: ",df_selected_control.count())

    rand_seed = 1

    print("rand_seed: ", rand_seed)

        # find those who did not match - join on just Age and Sex, smoking status
    df_unmatched_ids = df_patient_flags.join(df_selected_control.select("Patient_ID_"+cancer_site).distinct(), 
                                            how="leftanti",
                                            on="Patient_ID_"+cancer_site)
    
    if df_unmatched_ids.count()>0:
        all_ids_matched = False

    else:
        all_ids_matched = True

    i = 0 

    while all_ids_matched!= True and i<5: # keep iterating until all IDs matched to a control (max of 5 iterations)

        print("Iteration number for unmatched IDs: ", i+1)

        print("Number of unmatched IDs: ", df_unmatched_ids.count())

        # remove those IDs already used as controls
        df_multi_cohorts_exclude_existing_exclude_controls = df_multi_cohorts_exclude_existing.join(df_selected_control.select("Patient_ID").distinct(), 
                                                                                                    how="leftanti",
                                                                                                    on="Patient_ID")

        df_potential_controls_for_unmatched = df_multi_cohorts_exclude_existing_exclude_controls.join(df_unmatched_ids.select(["Patient_ID_"+cancer_site, "diagnosis_date_earliest"] + ["Cohort_Date", "AgeBand_5yr", "Sex", "Smoking_Flag"]),
                                                                                                    on = ["Cohort_Date", "AgeBand_5yr", "Sex", "Smoking_Flag"],
                                                                                                    how="inner")

        df_potential_controls_for_unmatched = df_potential_controls_for_unmatched.withColumn("rnd", F.rand(seed = rand_seed))

        window = Window.partitionBy("Patient_ID_" + cancer_site).orderBy(F.col("rnd"))

        df_selected_control_for_unmatched = df_potential_controls_for_unmatched.withColumn("row",F.row_number().over(window)) \
        .filter(F.col("row") == 1).drop("row")

        df_selected_control_for_unmatched= df_selected_control_for_unmatched.drop("rnd")

        df_selected_control_for_unmatched.cache()
        print("overall size of selected controls", df_selected_control_for_unmatched.count())

        number_unique_cases_being_matched_to = df_selected_control_for_unmatched.select("Patient_ID_" + cancer_site).distinct().count()
        number_unique_control_ids = df_selected_control_for_unmatched.select("Patient_ID").distinct().count()

        print("number_unique_cases_being_matched_to", number_unique_cases_being_matched_to)
        print("number_unique_control_ids", number_unique_control_ids)

        df_selected_control_for_unmatched = df_selected_control_for_unmatched.dropDuplicates(["Patient_ID"])

        print("Size of selected controls for those that were previously unmatched:", df_selected_control_for_unmatched.count())

        df_selected_control = df_selected_control.unionByName(df_selected_control_for_unmatched)
        
        df_selected_control.cache()


        df_unmatched_ids = df_patient_flags.join(df_selected_control.select("Patient_ID_"+cancer_site).distinct(), 
                                            how="leftanti",
                                            on="Patient_ID_"+cancer_site)
        
        if df_unmatched_ids.count() == 0:
            all_ids_matched = True

        i = i+1

    df_latest_cohort = df_selected_control

    df_latest_cohort = df_latest_cohort.withColumn(
        "history_start",
        F.date_sub(F.col("diagnosis_date_earliest"), 365)
    )

    df_latest_cohort.cache()
    print("Size of control dataset: ", df_latest_cohort.count())
    print("Number of unique lung cancer cases being matched to: ", df_latest_cohort.select("Patient_ID_" + cancer_site).distinct().count())
    print("Number of unique control IDs: ", df_latest_cohort.select("Patient_ID").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Events table

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cancer diagnosis event

# COMMAND ----------

if rerun_selection_controls == True:
    df_control_site = df_latest_cohort.select("Patient_ID", "diagnosis_date_earliest", "Age")
    df_control_site = df_control_site.withColumnRenamed("Age", "age_earliest_diagnosis")
    df_control_site = df_control_site.withColumn("dataset", F.lit("Cancer_registry"))
    df_control_site = df_control_site.withColumn("date", F.col("diagnosis_date_earliest"))
    df_control_site = df_control_site.withColumn("description", 
                            F.concat_ws("", F.lit("Date of diagnosis of matching case: "), F.col("diagnosis_date_earliest"),
                                        F.lit(" | ") , F.lit("Age at diagnosis: "), F.col("age_earliest_diagnosis")))
    
else:
    df_control_site = read_parquet_file(containerName =config.containerName_platinum, 
                                        lakeName=config.lakeName,
                                        filePath= f"")

    df_latest_cohort = read_parquet_file(containerName =config.containerName_platinum, 
                                        lakeName=config.lakeName,
                                        filePath= f"")

# COMMAND ----------

display(df_control_site)

# COMMAND ----------

df_control_site.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## GP events

# COMMAND ----------

df_pcp_emis = read_parquet_file(containerName =config.containerName_bronze, 
                             lakeName= config.lakeName,
                             filePath= config.filePath_emis) 

df_pcp_s1 = read_parquet_file(containerName =config.containerName_bronze, 
                           lakeName= config.lakeName,
                           filePath= config.filePath_s1)

df_gp_events = processing.process_gp_data_emis_s1(df_pcp_emis,df_pcp_s1 )

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
df_gp_events = df_gp_events.join(df_control_site.select("Patient_ID"), on = "Patient_ID", how="inner")

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

df_sus_all = df_sus_all.join(df_control_site.select("Patient_ID"),on = "Patient_ID", how="inner")
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

df_acute = df_acute.join(df_control_site.select("Patient_ID"),on = "Patient_ID", how="inner")
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

df_OoH = df_OoH.join(df_control_site.select("Patient_ID"), on = "Patient_ID", how="inner")
df_OoH = df_OoH.withColumn("dataset", F.lit("UC_OoH_All"))
df_OoH = df_OoH.withColumn("date", F.col("Attendance_Date"))

# COMMAND ----------

# remove duplicates
df_OoH = df_OoH.distinct()

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

df_999 = df_999.join(df_control_site.select("Patient_ID"),on = "Patient_ID", how="inner")
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

df_111 = df_111.join(df_control_site.select("Patient_ID"),on = "Patient_ID", how="inner")
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

df_gp_appointments = df_gp_appointments.join(df_control_site.select("Patient_ID"),on = "Patient_ID", how="inner")
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

# only keeping GP events for cancer patients
df_gp_meds = df_gp_meds.join(df_control_site.select("Patient_ID"), on = "Patient_ID", how="inner")

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
                            .unionByName(df_control_site.drop("diagnosis_date_earliest"), allowMissingColumns=True)\
                            .unionByName(df_ecds_data_chief_complaint, allowMissingColumns=True)\
                            .unionByName(df_gp_meds, allowMissingColumns=True)\
                            .unionByName(df_gp_appointments.drop("Decision_to_Refer_to_Service_Date", "Discharge_Date"), allowMissingColumns=True)

# COMMAND ----------

# add the cancer diagnosis date
df_all_activity = df_all_activity.join(df_control_site.select(["Patient_ID", "diagnosis_date_earliest"]), on = "Patient_ID", how="inner")
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

    df_control_site.write.mode("overwrite").parquet(
        fullPath_output,
    )

    fullPath_output=""

    df_all_activity.write.mode("overwrite").parquet(
        fullPath_output,
    )


