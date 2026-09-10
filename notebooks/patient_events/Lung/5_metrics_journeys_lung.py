# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 5. Lung Cancer Journey Metrics
# MAGIC This notebook creates statistics and visualisations showing relevant trends in patients' lung cancer diagnosis pathways. For example, plots are made highlighting delays between symptom reporting a chest x-rays being performed.
# MAGIC After the initial data import section has been run then each subsequent section can be run independently.

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import src.cancer_late.config  as config
# MAGIC import src.cancer_late.config_pathways  as config_pathways
# MAGIC from src.cancer_late.utils import read_parquet_file, read_csv_file
# MAGIC import src.cancer_late.patient_pathways_utils as patient_pathways_utils
# MAGIC from pyspark.sql import functions as F
# MAGIC from pyspark.sql.types import StructType, StructField, StringType, LongType
# MAGIC import matplotlib.pyplot as plt
# MAGIC import numpy as np
# MAGIC import seaborn as sns
# MAGIC from scipy.stats import skew, kurtosis, anderson_ksamp, kruskal
# MAGIC import pandas as pd
# MAGIC from operator import add
# MAGIC from functools import reduce
# MAGIC import plotly.express as px
# MAGIC import plotly.graph_objects as go
# MAGIC from src.cancer_late import processing
# MAGIC from pyspark.sql.window import Window
# MAGIC from scipy.stats import pearsonr

# COMMAND ----------

cancer_site = "Lung"
control_site = "Lung_control"
history_days = 365
version = config_pathways.cancer_site_mappings[cancer_site]["run_version"]

pal_dataset = {"ecds": "red", "gp": "green", "111":"orange", "acute":"purple", "overall": "grey"}
pal_stage = {"early":"green", "late": "red", "unknown": "grey"}
pal_imd = {"decile_1_to_3":"red", "decile_4_to_7": "orange" ,"decile_8_to_10": "green"}
pal_symptom =  {
    "cough": "tab:blue",
    "fatigue": "tab:orange",
    "shortness_breath": "tab:green",
    "chest_pain": "tab:red",
    "weight_loss": "tab:purple",
    "appetite_loss": "tab:brown",
    "chest_infection": "tab:pink",
    "finger_clubbing": "tab:gray",
    "supraclavicular_lymphadenopathy": "tab:olive",
    "cervical_lymphadenopathy": "tab:cyan",
    "thrombocytosis": "goldenrod",
    "haemoptysis": "darkgreen",
    "oral_antibiotic": "salmon",
    "inhaler": "mediumpurple",
}
list_of_mh_fields = ["LTC_MH", "LTC_Depression", "SMI_Personality_Disorder", "RF_CMHI_no_Depression", "RF_Self_Harm", "RF_Eating_Disorder", "LDA"]
var_mh = "flag_combined_mh" 

pal_route = {"Emergency Presentation": "blue", "Emergency presentation": "blue", "Unknown": "orange", "GP referral": "green", "Other outpatient": "red", "USC": "purple", "death_certificate_only": "brown", "Screening": "pink"}

markers = [
    "o",   # circle
    "s",   # square
    "D",   # diamond
    "^",   # triangle up
    "v",   # triangle down
    "<",   # triangle left
    ">",   # triangle right
    "P",   # plus-filled
    "X",   # x-filled
    "*",   # star
    "d",   # thin diamond
    "h",   # hexagon1
    "H",   # hexagon2
    "+",   # plus
    "x",   # x
    "|",   # vertical line
    "_",   # horizontal line
]

non_symptoms_111 = ['Healthcare_Professional_Callback',
                    'Covid_19_NHS_Pathway_response',
                    'Repeat_Prescription',
                    'Predetermined_Management_Plan',
                    'Non-trauma_Emergency',
                    'Unknown'
                    'NHS_Pathways_In_House_Clinician']

sources_of_flags = config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]

list_of_symptoms_incl_prescriptions = config_pathways.cancer_site_mappings["Lung"]["unexplained_symptoms_NICE_guidelines"] + config_pathways.cancer_site_mappings["Lung"]["critical_symptoms_NICE_guidelines"] + config_pathways.cancer_site_mappings["Lung"]["medication_flags"]

nice_guidelines_symptoms = config_pathways.cancer_site_mappings[cancer_site]["unexplained_symptoms_NICE_guidelines"] + config_pathways.cancer_site_mappings[cancer_site]["critical_symptoms_NICE_guidelines"]


cols_time_diagnosis_NG12 = ["time_diagnosis_for_smoker_with_unexplained_symptoms",
                            "time_diagnosis_for_two_or_more_unexplained_symptoms",
                            "time_diagnosis_for_one_critical_symptom",
                            f"days_xray_to_earliest_date_one_reported_unexplained_symptoms_NICE_guidelines_in_last_{history_days}_days",
                            f"days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_{history_days}_days",
                            f"days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_{history_days}_days",
                            f"days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_{history_days}_days",
                            "time_diagnosis_from_earliest_xray",
                            "time_diagnosis_from_latest_xray"]

def identify_val_in_list(col_name, list_of_items, null_return = None):
    for val in list_of_items:
        if val in col_name:
            return val
    return null_return


# COMMAND ----------

# MAGIC %md
# MAGIC # Import datasets

# COMMAND ----------

df_patient_flags = read_parquet_file(containerName =config.containerName_platinum, 
                                     lakeName=config.lakeName,
                                     filePath= f"")

df_pd_patient_flags = df_patient_flags.toPandas()
df_pd_patient_flags["category"] = "cases"
df_pd_patient_flags[var_mh] = np.where(df_pd_patient_flags[list_of_mh_fields].sum(axis=1)>=1, 1, 0)

df_patient_flags_control = read_parquet_file(containerName =config.containerName_platinum, 
                                     lakeName=config.lakeName,
                                     filePath= f"")

df_pd_patient_flags_control = df_patient_flags_control.toPandas()
df_pd_patient_flags_control["category"] = "controls"
df_pd_patient_flags_control[var_mh] = np.where(df_pd_patient_flags_control[list_of_mh_fields].sum(axis=1)>=1, 1, 0)


df_all_activity = read_parquet_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= f"")

# only consider activities at least one day before diagnosis
df_all_activity_before_diagnosis = df_all_activity.filter(F.col("days_between_activity_diagnosis")>=0)

df_all_activity_control = read_parquet_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= f"")

# only consider activities at least one day before diagnosis
df_all_activity_before_diagnosis_control = df_all_activity_control.filter(F.col("days_between_activity_diagnosis")>=0)

df_gp_red_flags = read_parquet_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= f"")

df_gp_red_flags_ref = read_parquet_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= f"")

df_referral = read_parquet_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= f"")

df_screening_snomed = read_parquet_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= f"")

df_acute_red_flag = read_parquet_file(containerName =config.containerName_platinum, 
                                    lakeName=config.lakeName,
                                    filePath= f"")

df_imd_gp = read_csv_file(containerName =config.containerName_platinum, 
                          lakeName=config.lakeName,
                          filePath= config_pathways.gp_deprivation).toPandas()

df_imd_gp_most_recent = df_imd_gp.sort_values(by=["Practice.Code","Year"], ascending=False).groupby("Practice.Code").first().reset_index()

# COMMAND ----------

list_red_flag_count_columns = [f"number_of_times_{source_of_flag}_red_flag_in_last_" + str(history_days) + "_days" for source_of_flag in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]] 
list_amber_flag_count_columns = [f"number_of_times_{source_of_flag}_amber_flag_in_last_" + str(history_days) + "_days" for source_of_flag in config_pathways.cancer_site_mappings[cancer_site]["sources_of_flags"]]

list_red_flag_count_columns.append("total_number_red_flag_symptoms_in_last_" + str(history_days) + "_days")
list_amber_flag_count_columns.append("total_number_amber_flag_symptoms_in_last_" + str(history_days) + "_days")

list_unique_datasets = df_all_activity_before_diagnosis.select("dataset").distinct().rdd.map(lambda dataset: dataset[0]).collect()
unique_days = np.linspace(0, history_days, history_days+1)
unique_ids = df_pd_patient_flags["Patient_ID"].unique()

date_columns_ng12 = ["earliest_date_one_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days",
                     "earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days",
                     "date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days",
                     "earliest_date_smoker_with_unexplained_symptoms_in_last_365_days",
                     "date_first_reported_chest_xray_in_last_365_days"]

flag_columns_ng12 = ["flag_smoker_with_unexplained_symptoms_in_last_365_days",
                     "flag_one_or_more_unexplained_symptoms_in_last_365_days",
                     "flag_two_or_more_unexplained_symptoms_in_last_365_days",
                     "flag_atleast_one_critical_symptoms_NICE_guidelines_in_last_365_days"]

date_symptoms_columns_to_process = []

for symptom in list_of_symptoms_incl_prescriptions:
    col_name = f"date_first_reported_{symptom}_in_last_{history_days}_days_overall"

    all_potential_columns = list(set(list(df_pd_patient_flags.columns) + list(df_pd_patient_flags_control.columns)))
    for col in all_potential_columns:
        if symptom in col and "date_first_reported" in col and "time_diagnosis" not in col: # ensure only dates are included
            date_symptoms_columns_to_process.append(col)
            print("processed", col)

date_columns_to_process = date_symptoms_columns_to_process + date_columns_ng12


# COMMAND ----------

# MAGIC %md
# MAGIC # Data quality checks

# COMMAND ----------

# MAGIC %md
# MAGIC ## check activity by patients

# COMMAND ----------

df_activity_per_dataset_per_patient = df_all_activity_before_diagnosis.filter(F.col("days_between_activity_diagnosis")<=history_days).groupBy(["Patient_ID"]).pivot("dataset").count().fillna(0).toPandas()
df_activity_per_dataset_per_patient = df_activity_per_dataset_per_patient.merge(df_pd_patient_flags[["Patient_ID", "Sex", "tumour_stage_group", "imd_decile_group", "Ethnic_Category", "age_10yr_band"]], on="Patient_ID", how="inner")
df_activity_per_dataset_per_patient

# COMMAND ----------

df_activity_per_dataset_per_patient.describe()

# COMMAND ----------

dict_dataset_count = {}

for dataset in list_unique_datasets:
    number_with_activity = df_activity_per_dataset_per_patient[df_activity_per_dataset_per_patient[dataset]!=0].shape[0]
    dict_dataset_count[dataset] = {100*number_with_activity/df_pd_patient_flags.shape[0]}
    print("Percentage who had activity in dataset", dataset, f"{100*number_with_activity/df_pd_patient_flags.shape[0]:.1f}", "%")

# COMMAND ----------

pd.DataFrame.from_dict(dict_dataset_count, orient="index")

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="Ethnic_Category", y="gp_events", hue="tumour_stage_group", data=df_activity_per_dataset_per_patient, hue_order = ["early", "late", "unknown"], palette = pal_stage)

plt.xticks(rotation=90);

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band", y="gp_events", hue = "tumour_stage_group" , data=df_activity_per_dataset_per_patient.sort_values(by="age_10yr_band"), hue_order = ["early", "late", "unknown"], palette = pal_stage)

plt.xticks(rotation=90);

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band", y="gp_events", hue = "Ethnic_Category" , data=df_activity_per_dataset_per_patient.sort_values(by="age_10yr_band"))

plt.xticks(rotation=90);

plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(x="Ethnic_Category", y="gp_events", hue="tumour_stage_group", data=df_activity_per_dataset_per_patient)

plt.xticks(rotation=90);

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(x="imd_decile_group", y="gp_events", hue="tumour_stage_group", data=df_activity_per_dataset_per_patient)

plt.xticks(rotation=90);

# COMMAND ----------

# MAGIC %md
# MAGIC ### number of events by EMIS or S1

# COMMAND ----------

df_gp_activity_with_practice = df_all_activity_before_diagnosis.filter((F.col("dataset")=="gp_events") &
                                                    (F.col("days_between_activity_diagnosis")<=history_days)).drop("Practice_Code").join(df_patient_flags.select(["Patient_ID", "Practice_Code"]), on="Patient_ID", how="inner")


df_gp_activity_with_practice_controls = df_all_activity_before_diagnosis_control.filter((F.col("dataset")=="gp_events") &
                                                                                (F.col("days_between_activity_diagnosis")<=history_days)).drop("Practice_Code").join(df_patient_flags_control.select(["Patient_ID", "Practice_Code"]), on="Patient_ID", how="inner")

df_num_events_by_gp_source = df_gp_activity_with_practice.groupby(["Patient_ID"]).pivot("source").count().toPandas()
df_num_events_by_gp_source

# COMMAND ----------

df_num_events_by_gp_source[df_num_events_by_gp_source["s1"].isnull()]["emis"].describe()

# COMMAND ----------

df_num_events_by_gp_source[df_num_events_by_gp_source["emis"].isnull()]["s1"].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check activity by practice

# COMMAND ----------

df_patient_flags.groupby("Practice_Code").count().display()

# COMMAND ----------

# number of events, per patient, per practice
df_gp_activity_by_patient_practice = df_gp_activity_with_practice.groupby(["Patient_ID", "Practice_Code"]).count()

df_gp_activity_by_patient_practice_controls = df_gp_activity_with_practice_controls.groupby(["Patient_ID", "Practice_Code"]).count()



display(df_gp_activity_by_patient_practice)

# COMMAND ----------

df_practice_stats = df_gp_activity_by_patient_practice.groupby("Practice_Code").agg(
    F.count("count").alias("count"),
    F.median("count").alias("median"),
    F.mean("count").alias("mean"),
    F.stddev("count").alias("std"),
    F.min("count").alias("min"),
    F.max("count").alias("max"),
)

df_practice_stats_pd = df_practice_stats.toPandas()

df_practice_stats_controls = df_gp_activity_by_patient_practice_controls.groupby("Practice_Code").agg(
    F.count("count").alias("count"),
    F.median("count").alias("median"),
    F.mean("count").alias("mean"),
    F.stddev("count").alias("std"),
    F.min("count").alias("min"),
    F.max("count").alias("max"),
)

df_practice_stats_controls_pd = df_practice_stats_controls.toPandas()

print("Size of df_practice_stats_pd: ", df_practice_stats_pd.shape)

display(df_practice_stats)

# COMMAND ----------

ax = plt.figure(figsize=(10,6))
sns.set(font_scale = 1, style="white")

ax = sns.histplot(data = df_practice_stats_pd, x="median", bins = 15, kde=True, label = "cases")
ax = sns.histplot(data = df_practice_stats_controls_pd, x="median", bins = 15, kde=True, label = "controls")

plt.xlabel("Median number of events per patient in the last year before cancer diagnosis")
plt.ylabel("Number of practices")
plt.legend()

# COMMAND ----------

df_practice_stats_cases_controls = df_practice_stats_pd.merge(df_practice_stats_controls_pd, on = "Practice_Code", how="inner", suffixes = ("_cases", "_controls"))
df_practice_stats_cases_controls

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.set(font_scale = 1.5, style="white")

sns.scatterplot(data = df_practice_stats_cases_controls, x="median_cases", y="median_controls")

plt.ylabel("Median number of events per patient (control)")
plt.xlabel("Median number of events per patient (cases)")

plt.xlim(left = 0)
plt.ylim(bottom = 0)

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.set(font_scale = 1.5, style="white")

sns.scatterplot(data = df_practice_stats_pd, x="count", y="median")

plt.ylabel("Median number of events per patient")
plt.xlabel("Number of patients in practice")

# COMMAND ----------

# MAGIC %md
# MAGIC ### number of red flag symptoms by practice

# COMMAND ----------

# red flag symptoms as proportion of total number of codes
# referral as proportion of total number of codes

df_gp_activity_with_practice_join_red_flag_symptoms = df_gp_activity_with_practice.join(df_gp_red_flags_ref.drop("term"),
                                                                                        df_gp_activity_with_practice.Concept_ID == df_gp_red_flags_ref.code,
                                                                                        how = 'left')

df_gp_red_flag_by_patient_practice = df_gp_activity_with_practice_join_red_flag_symptoms.filter(F.col("grouping")=="red_flag").groupby(["Patient_ID", "Practice_Code", "grouping"]).count()
df_gp_red_flag_by_patient_practice = df_gp_red_flag_by_patient_practice.withColumnRenamed("count", "red_flag_count")

df_pd_red_flag_activity_per_patient = df_gp_activity_by_patient_practice.join(df_gp_red_flag_by_patient_practice.select(["Patient_ID", "red_flag_count"]),
                                                                           on="Patient_ID",
                                                                           how="left").fillna(0).toPandas()

df_pd_red_flag_activity_per_patient["any_red_flag"] = np.where(df_pd_red_flag_activity_per_patient["red_flag_count"] > 0, "red_flag_present", "no_red_flag")
df_pd_red_flag_activity_per_patient = df_pd_red_flag_activity_per_patient.drop(columns = ["Practice_Code"]).merge(df_pd_patient_flags, on ="Patient_ID", how="inner")

df_pd_red_flag_activity_per_patient_filtered_by_route = df_pd_red_flag_activity_per_patient[df_pd_red_flag_activity_per_patient["route_earliest"].isin(["USC", "GP referral"])]

df_percent_stage_by_practice = patient_pathways_utils.count_and_pct(df_pd_red_flag_activity_per_patient_filtered_by_route, "tumour_stage_group", "Practice_Code" ).pivot_table(index="Practice_Code", columns = "tumour_stage_group", values =  "percentage").fillna(0).reset_index()

df_pd_red_flag_activity_per_practice = patient_pathways_utils.count_and_pct(df_pd_red_flag_activity_per_patient_filtered_by_route, "any_red_flag", "Practice_Code" ).pivot_table(index="Practice_Code", columns = "any_red_flag", values = ["count", "percentage"]).fillna(0).reset_index()

df_pd_red_flag_activity_per_practice["total_count"] = df_pd_red_flag_activity_per_practice["count"]["no_red_flag"] + df_pd_red_flag_activity_per_practice["count"]["red_flag_present"]

new_cols = []

for col in df_pd_red_flag_activity_per_practice.columns:

    if len(col[1]) > 0:
        new_cols.append(col[0]+"_"+col[1])
    else:
        new_cols.append(col[0])

df_pd_red_flag_activity_per_practice.columns = new_cols

df_pd_red_flag_activity_per_practice_wth_pct_stage = df_pd_red_flag_activity_per_practice.merge(df_percent_stage_by_practice, on="Practice_Code")

plt.figure(figsize=(8,8))
sns.set(font_scale = 1, style="white")

sns.scatterplot(x="total_count", y="percentage_red_flag_present", data = df_pd_red_flag_activity_per_practice_wth_pct_stage, hue = "early")

plt.title("Percentage of patients with reported red flag symptom - only patients where route to diagnosis was GP referral/TWW/USC")
plt.xlabel("Total number in practice")
plt.ylabel("Percentage of patients in practice with red flag symptom reported")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2, title = "percentage with early stage diagnosis")

# COMMAND ----------

df_pd_red_flag_activity_per_patient_filtered_by_route = df_pd_red_flag_activity_per_patient[df_pd_red_flag_activity_per_patient["route_earliest"].isin(["GP referral"])]

df_percent_stage_by_practice = patient_pathways_utils.count_and_pct(df_pd_red_flag_activity_per_patient_filtered_by_route, "tumour_stage_group", "Practice_Code" ).pivot_table(index="Practice_Code", columns = "tumour_stage_group", values =  "percentage").fillna(0).reset_index()

df_pd_red_flag_activity_per_practice = patient_pathways_utils.count_and_pct(df_pd_red_flag_activity_per_patient_filtered_by_route, "any_red_flag", "Practice_Code" ).pivot_table(index="Practice_Code", columns = "any_red_flag", values = ["count", "percentage"]).fillna(0).reset_index()

df_pd_red_flag_activity_per_practice["total_count"] = df_pd_red_flag_activity_per_practice["count"]["no_red_flag"] + df_pd_red_flag_activity_per_practice["count"]["red_flag_present"]

new_cols = []

for col in df_pd_red_flag_activity_per_practice.columns:

    if len(col[1]) > 0:
        new_cols.append(col[0]+"_"+col[1])
    else:
        new_cols.append(col[0])

df_pd_red_flag_activity_per_practice.columns = new_cols

df_pd_red_flag_activity_per_practice_wth_pct_stage = df_pd_red_flag_activity_per_practice.merge(df_percent_stage_by_practice, on="Practice_Code")

plt.figure(figsize=(8,8))
sns.set(font_scale = 1, style="white")

sns.scatterplot(x="total_count", y="percentage_red_flag_present", data = df_pd_red_flag_activity_per_practice_wth_pct_stage, hue = "early")

plt.title("Percentage of patients with reported red flag symptom - only patients where route to diagnosis was USC")
plt.xlabel("Total number in practice")
plt.ylabel("Percentage of patients in practice with red flag symptom reported")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2, title = "percentage with early stage diagnosis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Number of chest x-rays by GP practice

# COMMAND ----------

df_cases_and_controls = pd.concat([df_pd_patient_flags, df_pd_patient_flags_control], axis = 0).copy()
df_number_xrays_by_practice = df_cases_and_controls.groupby(["category","Practice_Code"]).agg(number_patients = ("Patient_ID","count"), number_patients_with_xray = ("flag_chest_xray_in_last_365_days", "sum")) 
df_number_xrays_by_practice.columns = ["number_patients", "number_patients_with_xray"]
df_number_xrays_by_practice = df_number_xrays_by_practice.reset_index()
df_number_xrays_by_practice = df_number_xrays_by_practice.pivot_table(index = "Practice_Code", columns = "category", values = ["number_patients", "number_patients_with_xray"]).fillna(0).reset_index()
df_number_xrays_by_practice["pct_cases"] = 100*df_number_xrays_by_practice["number_patients_with_xray"]["cases"]/df_number_xrays_by_practice["number_patients"]["cases"]
df_number_xrays_by_practice["pct_controls"] = 100*df_number_xrays_by_practice["number_patients_with_xray"]["controls"]/df_number_xrays_by_practice["number_patients"]["controls"]
new_cols = []
for col in df_number_xrays_by_practice.columns:
    if len(col[1])>0:
        new_col = col[0] + "_" + col[1]
    else:
        new_col = col[0]
    
    new_cols.append(new_col)

df_number_xrays_by_practice.columns = new_cols

# COMMAND ----------

df_describe_xray_by_gp = df_number_xrays_by_practice.describe()

# find 25th percentile, median, and 75th percentile of percentage of cases with chest imaging
pct_25_val =  df_describe_xray_by_gp.loc["25%", "pct_cases"]
pct_50_val =  df_describe_xray_by_gp.loc["50%", "pct_cases"]
pct_75_val =  df_describe_xray_by_gp.loc["75%", "pct_cases"]

df_describe_xray_by_gp 


# COMMAND ----------

df_num_xrays_per_practice = df_pd_patient_flags.groupby("Practice_Code")["number_of_times_chest_xray_in_last_365_days"].describe().reset_index()
df_time_to_diagnosis_from_earliest_xray = df_pd_patient_flags.groupby("Practice_Code")["time_diagnosis_from_earliest_xray"].describe().reset_index()
df_practice_xray_stats = df_num_xrays_per_practice.merge(df_time_to_diagnosis_from_earliest_xray, on = "Practice_Code", how="inner", suffixes = ("_num_xrays", "_time_diagnosis_from_earliest_xray"))
df_practice_xray_stats = df_practice_xray_stats.merge(df_number_xrays_by_practice, on = "Practice_Code")

df_practice_stage = df_pd_patient_flags.groupby("Practice_Code")["tumour_stage_group"].value_counts().to_frame()
df_practice_stage.columns = ["count"]
df_practice_stage = df_practice_stage.reset_index().pivot_table(index="Practice_Code", columns = "tumour_stage_group", values = "count").reset_index().fillna(0)
 
df_practice_deprivation = df_pd_patient_flags.groupby("Practice_Code")["imd_decile_group"].value_counts().to_frame()
df_practice_deprivation.columns = ["count"]
df_practice_deprivation = df_practice_deprivation.reset_index().pivot_table(index="Practice_Code", columns = "imd_decile_group", values = "count").reset_index().fillna(0)
 
df_practice_stage_deprivation = df_practice_stage.merge(df_practice_deprivation, on = "Practice_Code", how="inner")
 
df_practice_xray_stats = df_practice_xray_stats.merge(df_practice_stage_deprivation, on = "Practice_Code", how="inner")
df_practice_xray_stats["pct_early_stage"] =  100*df_practice_xray_stats["early"]/df_practice_xray_stats["number_patients_cases"]
df_practice_xray_stats["pct_late_stage"] =  100*df_practice_xray_stats["late"]/df_practice_xray_stats["number_patients_cases"]
df_practice_xray_stats["pct_least_deprived"] =  100*df_practice_xray_stats["decile_1_to_3"]/df_practice_xray_stats["number_patients_cases"]
df_practice_xray_stats["pct_most_deprived"] =  100*df_practice_xray_stats["decile_8_to_10"]/df_practice_xray_stats["number_patients_cases"]
df_practice_xray_stats = df_practice_xray_stats.merge(df_imd_gp_most_recent, left_on = "Practice_Code", right_on = "Practice.Code", how="left")

df_practice_xray_stats

# COMMAND ----------

df_describe_xray_stats = df_practice_xray_stats.describe()
df_describe_xray_stats

# COMMAND ----------

n_size_practices = 10

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

df = df_practice_xray_stats
df=df[df["number_patients_cases"]>=n_size_practices]

x = df[~df["pct_cases"].isnull()]["number_patients_cases"]
y = df[~df["pct_cases"].isnull()]["pct_cases"]

sns.regplot(x=x, y=y, data = df_practice_xray_stats)
plt.xlabel("Number of cancer cases in GP practice")
plt.ylabel("Percentage of cases with an xray")

pearsonr(x, y)


# COMMAND ----------

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

df = df_practice_xray_stats
df=df[df["number_patients_controls"]>=n_size_practices]

x = df[~df["pct_controls"].isnull()]["number_patients_controls"]
y = df[~df["pct_controls"].isnull()]["pct_controls"]


sns.regplot(x=x, y=y)
plt.xlabel("Number of control cases in GP practice")
plt.ylabel("Percentage of controls with an xray")

pearsonr(x, y)


# COMMAND ----------

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

sns.scatterplot(x="number_patients_with_xray_cases", y="number_patients_with_xray_controls", data = df_practice_xray_stats)
plt.xlabel("Number of cancer cases with chest imaging in 1 year before in GP practice")
plt.ylabel("Number of controls with chest imaging in 1 year before in GP practice")

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

sns.regplot(x="pct_cases", y="pct_controls", data = df_practice_xray_stats)
plt.xlabel("Percentage of cancer cases with chest imaging in 1 year before")
plt.ylabel("Percentage of controls with chest imaging in 1 year before")

# COMMAND ----------

df = df_practice_xray_stats.copy()
df = df[(df["number_patients_cases"]>=n_size_practices)]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1.25, style="white")

sns.regplot(x="pct_cases", y="pct_controls", data = df)
plt.xlabel("Percentage of cancer cases with chest imaging in 1 year before")
plt.ylabel("Percentage of controls with chest imaging in 1 year before")

pearsonr(df["pct_cases"], df["pct_controls"])

# COMMAND ----------

# find practices with high % xrays in controls and those with low
# find % early and late

n_size_practices = 10

pct_75_controls = df_practice_xray_stats[df_practice_xray_stats["number_patients_controls"]>=n_size_practices].describe().loc["75%","pct_controls"]
pct_25_controls = df_practice_xray_stats[df_practice_xray_stats["number_patients_controls"]>=n_size_practices].describe().loc["25%","pct_controls"] 

gp_practices_few_xrays = list(df_practice_xray_stats[(df_practice_xray_stats["number_patients_controls"]>=n_size_practices) & (df_practice_xray_stats["pct_controls"]<=pct_25_controls)]["Practice_Code"])
gp_practices_more_xrays = list(df_practice_xray_stats[(df_practice_xray_stats["number_patients_controls"]>=n_size_practices) & (df_practice_xray_stats["pct_controls"]>=pct_75_controls)]["Practice_Code"])

df = df_practice_xray_stats[df_practice_xray_stats["number_patients_cases"]>=n_size_practices].copy()

df_low = df_pd_patient_flags[df_pd_patient_flags["Practice_Code"].isin(gp_practices_few_xrays)].copy()
df_low["xrays_relative"] = "fewer_xrays"

df_high = df_pd_patient_flags[df_pd_patient_flags["Practice_Code"].isin(gp_practices_more_xrays)].copy()
df_high["xrays_relative"] = "more_xrays"

df = pd.concat([df_low, df_high], axis = 0)

patient_pathways_utils.count_and_pct(df,  col = "tumour_stage_group", groupby_col= "xrays_relative")




# COMMAND ----------

# plot avg number of xrays per patient, by avg time between earliest xray and diagnosis

df = df_practice_xray_stats.copy()
df = df[(df["number_patients_cases"]>=n_size_practices)]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

sns.regplot(x="mean_num_xrays", y="pct_early_stage", data = df)

plt.xlabel("Mean number of xrays per patient")
plt.ylabel("Percentage with early lung cancer diagnosis")
plt.title(f"Only including GP practices with at least {n_size_practices} cancer cases");

# COMMAND ----------

# plot avg number of xrays per patient, by avg time between earliest xray and diagnosis

df = df_practice_xray_stats.copy()
df = df[(df["number_patients_cases"]>=n_size_practices)]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

sns.regplot(x="mean_num_xrays", y="pct_late_stage", data = df)

plt.xlabel("Mean number of xrays per patient")
plt.ylabel("Percentage with late lung cancer diagnosis")

# COMMAND ----------

df = df_practice_xray_stats.copy()
df = df[df["number_patients_cases"]>=n_size_practices]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

sns.regplot(x="mean_time_diagnosis_from_earliest_xray", y="pct_early_stage", data = df)

plt.xlabel("Mean number of days between earliest xray and diagnosis")
plt.ylabel("Percentage with early lung cancer diagnosis")


# COMMAND ----------

df = df_practice_xray_stats.copy()
df = df[df["number_patients_cases"]>=n_size_practices]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

sns.regplot(x="mean_time_diagnosis_from_earliest_xray", y="pct_late_stage", data = df)

plt.xlabel("Mean number of days between earliest xray and diagnosis")
plt.ylabel("Percentage with late lung cancer diagnosis")


# COMMAND ----------

df = df_practice_xray_stats.copy()
df = df[df["number_patients_cases"]>=n_size_practices]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1, style="white")

sns.scatterplot(x="mean_time_diagnosis_from_earliest_xray", y="mean_num_xrays", hue = "pct_late_stage", data = df)


# COMMAND ----------

# MAGIC %md
# MAGIC ### by deprivation

# COMMAND ----------

df = df_practice_xray_stats.copy()
df = df[(df["number_patients_cases"]>=n_size_practices)]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1.25, style="white")

sns.regplot(x="IMD", y="pct_cases", data = df)
plt.xlabel("Deprivation score for GP practice")
plt.ylabel("Percentage of cases with chest imaging in 1 year before")

pearsonr(df["IMD"], df["pct_cases"])

# COMMAND ----------

df = df_practice_xray_stats.copy()
df = df[(df["number_patients_cases"]>=n_size_practices)]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1.5, style="white")

sns.regplot(x="IMD", y="mean_num_xrays", data = df)
plt.xlabel("Deprivation score for GP practice")
plt.ylabel("Number of xrays per patient")

pearsonr(df["IMD"], df["mean_num_xrays"])

# COMMAND ----------

df = df_practice_xray_stats.copy()
df = df[(df["number_patients_cases"]>=n_size_practices)]

plt.figure(figsize=(10,10))
sns.set(font_scale = 1.5, style="white")

sns.regplot(x="IMD", y="pct_controls", data = df)
plt.xlabel("Deprivation score for GP practice")
plt.ylabel("Percentage of controls with chest imaging in 1 year before")

pearsonr(df["IMD"], df["pct_controls"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Overall metrics

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="YearMonth", y="count", data = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "YearMonth"))

plt.xticks(rotation=90);

# COMMAND ----------

# MAGIC %md
# MAGIC ## sex

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "Sex")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags,  col = "tumour_stage_group", groupby_col= "Sex")

# COMMAND ----------

plt.figure(figsize=(6,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="Sex",
            y="percentage",
            hue = "tumour_stage_group",
            hue_order = ["early", "late", "unknown"],
            palette = pal_stage,
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags,  col = "tumour_stage_group", groupby_col= "Sex")
)
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ethnicity

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "Ethnic_Category")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "Ethnic_Category", "Age")

# COMMAND ----------

df_pd_patient_flags.groupby("Ethnic_Category")["Age"].describe()

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col="Ethnic_Category")

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="Ethnic_Category",
            y="count",
            hue = "tumour_stage_group",
            hue_order = ["early", "late", "unknown"],
            palette = pal_stage,
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "Ethnic_Category")
)
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1.5, style="white")

sns.barplot(y="Ethnic_Category",
            x="percentage",
            hue = "tumour_stage_group",
            hue_order = ["early", "late", "unknown"],
            palette = pal_stage,
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "Ethnic_Category")
)
plt.xticks(rotation=90);
plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## smoking status

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "Smoking_Flag")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags_control, "Smoking_Flag")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col="Smoking_Flag")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col ="Smoking_Flag" , groupby_col="route_earliest")

# COMMAND ----------

# for those without smoking status of 1

patient_pathways_utils.count_and_pct(df_pd_patient_flags[df_pd_patient_flags["Smoking_Flag"]!=1], "tumour_stage_group")

# COMMAND ----------

print("percentage of smokers in lung cancer population: ", 100*df_pd_patient_flags[df_pd_patient_flags["Smoking_Flag"]==1].shape[0]/df_pd_patient_flags.shape[0])
print("percentage of smokers in control population: ", 100*df_pd_patient_flags_control[df_pd_patient_flags_control["Smoking_Flag"]==1].shape[0]/df_pd_patient_flags_control.shape[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## stage

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "tumour_stage_group")

# COMMAND ----------

df_stages_by_month = patient_pathways_utils.count_and_pct(df_pd_patient_flags, col= "tumour_stage_group", groupby_col="YearMonth" )

plt.figure(figsize=(18,6))
sns.set(font_scale = 1, style="white")

sns.lineplot(x="YearMonth", y="percentage", hue="tumour_stage_group", data = df_stages_by_month)

plt.xticks(rotation=90);
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## deprivation 

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "imd_decile_group")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags_control, "imd_decile_group")

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1.5, style="white")

df = patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "imd_decile_group")
df = df[df["imd_decile_group"]!="unknown"]

sns.barplot(x="imd_decile_group",
            y="percentage",
            hue = "tumour_stage_group",
            hue_order = ["early", "late", "unknown"],
            palette = pal_stage,
            data=df
)
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

fig, ax = plt.subplots(1,1,figsize=(10,5))

offset = -0.2
for name, dataset in {"Control": df_patient_flags, "Cancer": df_patient_flags_control}.items():
    dataset = dataset.filter(F.col("IMD_Decile").isNotNull())
    imd_count_df = dataset.select("Patient_ID", "IMD_Decile").distinct().groupBy("IMD_Decile").count().toPandas()
    x = imd_count_df["IMD_Decile"].astype(int)
    x_offset = [i + offset for i in x]
    heights = 100*imd_count_df["count"]/imd_count_df["count"].sum()
    ax.bar(x_offset, heights, label=name, width=0.4)
    offset +=0.4

ax.set_xticks(range(1,11))
ax.set_ylabel("% of cohort")
ax.set_xlabel("IMD Decile")
ax.set_title(f"IMD Distribution of Lung Cancer and LDM Control Cohort")
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
plt.show()

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "tumour_stage_group", "imd_decile_group")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "tumour_stage_group", "IMD_Decile")

# COMMAND ----------

# MAGIC %md
# MAGIC ## age

# COMMAND ----------

df_pd_patient_flags["Age"].describe()

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "age_10yr_band")

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band", y="count", data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, "age_10yr_band"))
plt.xticks(rotation=90);

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "age_10yr_band")

# COMMAND ----------


plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band",
            y="percentage",
            hue = "tumour_stage_group",
            hue_order = ["early", "late", "unknown"],
            palette = pal_stage,
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "age_10yr_band")
)
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band",
            y="count",
            hue = "tumour_stage_group",
            hue_order = ["early", "late", "unknown"],
            palette = pal_stage,
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "age_10yr_band")
)
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band",
            y="count",
            hue = "tumour_stage_group",
            hue_order = ["early", "late", "unknown"],
            palette = pal_stage,
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "age_10yr_band")
)
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

df_pd_patient_flags["age_74"] = np.where(df_pd_patient_flags["Age"]>74, ">74", "<=74")

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "age_74")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Age and deprivation (% early stage)
# MAGIC

# COMMAND ----------

df_pd_patient_flags.groupby(["age_10yr_band", "imd_decile_group"])["tumour_stage_group"].value_counts().to_frame().iloc[0:20]

# COMMAND ----------


df = df_pd_patient_flags.groupby(["age_10yr_band", "imd_decile_group"])["tumour_stage_group"].value_counts(normalize=True).to_frame()
#.reset_index()
df.columns = ["Percentage"]
df["Percentage"] = df["Percentage"] *100 
df = df.reset_index()

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")
                                

sns.barplot(x="age_10yr_band",
            y="Percentage",
            hue = "imd_decile_group",
            data= df[(df["tumour_stage_group"]=="early") & (df["age_10yr_band"].isin(["30-39", "40-49", "50-59", "60-69", "70-79", "80-89" ]))
            & (df["imd_decile_group"]!="unknown")]
)
plt.xticks(rotation=90);
plt.ylabel("Percentage diagnosed early (%)")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Route to diagnosis

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags,
                                     col = "route_earliest").sort_values(by="percentage", ascending=False)

# COMMAND ----------

route_order = list(patient_pathways_utils.count_and_pct(df_pd_patient_flags,col = "route_earliest").sort_values(by="percentage", ascending=True)["route_earliest"])

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1.5, style="white")

sns.barplot( y="percentage",
            x = "route_earliest",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col ="route_earliest"),
            order = route_order,
)
plt.xticks(rotation=90);
plt.ylabel("Percentage")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags,
                                     col = "tumour_stage_group",
                                     groupby_col="route_earliest")

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1.5, style="white")

df = patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col= "route_earliest")
df = df[~df["route_earliest"].isin(["Screening", "death_certificate_only", "Unknown"])]

sns.barplot(hue="tumour_stage_group",
            y="percentage",
            x = "route_earliest",
            data= df
)
plt.xticks(rotation=90);
plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

value = "route_earliest"
val_count_df = df_patient_flags.select("Patient_ID", value, "tumour_stage_group").distinct().groupBy(value, "tumour_stage_group").count().toPandas()

val_count_df = val_count_df.replace({"death_certificate_only": "Death Certificate Only"})

fig, ax = plt.subplots(1,1,figsize=(8,5))
val_count_df["% of population"] = 100*val_count_df["count"]/val_count_df["count"].sum()
total_count_df = val_count_df.groupby("route_earliest").sum()
total_count_df = total_count_df.sort_values(by="% of population")
total_count_df = total_count_df.replace({"death_certificate_only": "Death Certificate Only"})
total_x = total_count_df.index
total_y = total_count_df["% of population"]

ax.barh(total_x, total_y, label="unknown", color="grey")

early_count_df = val_count_df[val_count_df["tumour_stage_group"].isin(["early", "late"])].groupby("route_earliest").sum()
early_count_df = early_count_df.sort_values(by="% of population")
early_x = early_count_df.index
early_y = early_count_df["% of population"]

ax.barh(early_x, early_y, label="early", color="green")

late_count_df = val_count_df[val_count_df["tumour_stage_group"] == "late"]
late_count_df = late_count_df.sort_values(by="count")
late_x = late_count_df[value]
late_y = late_count_df["% of population"]

route_count_df = patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest").sort_values(by="count", ascending=True)
route_count_dict = pd.Series(route_count_df["count"].values, index=route_count_df["route_earliest"]).to_dict()
routes = list(route_count_df["route_earliest"])
y_tick_labels = [f"{route} (n={route_count_dict[route]})" for route in routes]


ax.barh(late_x, late_y, label="late", color="red")
ax.set_title(f"Routes to Lung Cancer Diagnosis by Stage")
ax.set_ylabel("Routes to diagnosis")
ax.set_xlabel("% of cancer cohort")
ax.set_yticklabels(y_tick_labels)
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
handles, labels = ax.get_legend_handles_labels()
order = [1,2,0]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.show()

# COMMAND ----------

imd_route_df = df_patient_flags.groupBy("imd_decile_group", "route_earliest").count().toPandas()
imd_route_df = imd_route_df.sort_values(by=["imd_decile_group", "route_earliest"], ascending=False)

imds = imd_route_df["imd_decile_group"].unique()

fig, ax = plt.subplots(1,1,figsize=(8,6))
offset = -0.2
for imd in imds:
    if imd != "unknown":
        imd_df = imd_route_df[imd_route_df["imd_decile_group"] == imd].sort_values(by="count")
        y_tick_labels = imd_df["route_earliest"]
        imd_df["percent"] = (100/imd_df["count"].sum())*imd_df["count"]
        widths = imd_df["count"]
        y_ticks = [i+offset for i in range(len(y_tick_labels))]
        offset += 0.2
        ax.barh(y_ticks, widths, label=imd, height=0.2)
    
ax.set_title(f"Lung Cancer Diagnosis Route by Patient IMD")
ax.set_xlabel("no. of patients")
ax.set_yticks(range(len(y_tick_labels)), labels=y_tick_labels)
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
plt.show()

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags,
                                     col = "LTC_COPD",
                                     groupby_col="route_earliest")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags,
                                     col = "imd_decile_group",
                                     groupby_col="route_earliest")

# COMMAND ----------

df_pd_patient_flags.groupby("route_earliest")["Age"].describe()

# COMMAND ----------

df_pd_patient_flags.groupby("route_earliest")["Sex"].value_counts(normalize=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## deprivation

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags,
                                     col = "imd_decile_group",
                                     groupby_col="route_earliest").sort_values(["route_earliest", "imd_decile_group"])

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="imd_decile_group",
            y="percentage",
            hue = "route_earliest",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest", groupby_col= "imd_decile_group")
)
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## age

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest", groupby_col="age_10yr_band")

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1.5, style="white")

sns.barplot(x="age_10yr_band",
            y="percentage",
            hue = "route_earliest",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest", groupby_col= "age_10yr_band"),
            hue_order = route_order,
)
plt.xticks(rotation=90);
plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band",
            y="count",
            hue = "route_earliest",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest", groupby_col= "age_10yr_band"),
            hue_order = route_order,
)
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ethnicity

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest", groupby_col="Ethnic_Category")

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="Ethnic_Category",
            y="percentage",
            hue = "route_earliest",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest", groupby_col= "Ethnic_Category")
)
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## COPD Diagnosis

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, "LTC_COPD")

# COMMAND ----------

df_stage_by_copd = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "tumour_stage_group", "LTC_COPD")


sns.barplot(hue="tumour_stage_group",
            y="percentage",
            x = "LTC_COPD",
            data= df_stage_by_copd,
            palette = pal_stage
)

plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

display(df_stage_by_copd)

# COMMAND ----------

df_route_by_copd = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "route_earliest", "LTC_COPD")

sns.barplot(hue="route_earliest",
            y="percentage",
            x = "LTC_COPD",
            data= df_route_by_copd
)

plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

df_patient_flags = df_patient_flags.withColumn("copd_temp", F.when(F.col("LTC_COPD") == 1, True).otherwise(False))
pivot_df = df_patient_flags.groupBy("route_earliest", "copd_temp").count().groupBy("route_earliest").pivot("copd_temp").sum().filter(F.col("route_earliest").isNotNull())
pivot_df = pivot_df.withColumn("copd_percent", 100*F.col("true")/(F.col("true") + F.col("false")))
smoker_df = pivot_df.select("route_earliest", "copd_percent").orderBy("copd_percent").toPandas()

fig, ax = plt.subplots(1,1,figsize=(10,5))

widths = smoker_df["copd_percent"].values
y_ticks = smoker_df["route_earliest"].values
ax.barh(y_ticks, widths)
ax.set_title("% of Patients with COPD Flag by Diagnosis Route")
ax.set_yticks(range(len(y_ticks)), labels=y_ticks)
ax.set_xlabel("% of patients")

# COMMAND ----------

pivot_df = df_patient_flags.groupBy("IMD_Decile", "copd_temp").count().groupBy("IMD_Decile").pivot("copd_temp").sum().filter(F.col("IMD_Decile").isNotNull())
pivot_df = pivot_df.withColumn("copd_percent", 100*F.col("true_sum(count)")/(F.col("true_sum(count)") + F.col("false_sum(count)")))
smoker_df = pivot_df.select("IMD_Decile", "copd_percent").orderBy("copd_percent").toPandas()

fig, ax = plt.subplots(1,1,figsize=(10,5))

widths = smoker_df["copd_percent"].values
y_ticks = smoker_df["IMD_Decile"].values
ax.bar(y_ticks, widths)
ax.set_title("% of Patients with COPD Flag by IMD")
ax.set_xticks(range(1,11))
ax.set_xlabel("IMD Decile")
ax.set_ylabel("% of patients")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mental Health Illness

# COMMAND ----------

df_mh_metrics = pd.DataFrame()

for metric in list_of_mh_fields + ["Frailty_level" , var_mh]:

    df_case_pct = patient_pathways_utils.count_and_pct(df_pd_patient_flags, metric).rename(columns = {"percentage": "Cancer cohort (%)"}).drop(columns = ["count"])
    df_control_pct = patient_pathways_utils.count_and_pct(df_pd_patient_flags_control, metric).rename(columns = {"percentage": "Control (%)"}).drop(columns = ["count"])
    df_case_stage_pct = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "tumour_stage_group", metric)
    df_case_stage_pct = df_case_stage_pct.pivot_table(index=metric, columns = "tumour_stage_group", values = "percentage").reset_index()
    
    df_case_route_pct = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "route_earliest", metric)
    df_case_route_pct = df_case_route_pct.pivot_table(index=metric, columns = "route_earliest", values = "percentage").reset_index()
    
    df_case_deprivation_pct = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "imd_decile_group", metric)
    df_case_deprivation_pct = df_case_deprivation_pct.pivot_table(index=metric, columns = "imd_decile_group", values = "percentage").reset_index()
    
    df_case_smoking_pct = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "Smoking_Flag", metric)
    df_case_smoking_pct= df_case_smoking_pct[df_case_smoking_pct["Smoking_Flag"]==True].reset_index(drop=True).drop(columns = ["count","Smoking_Flag"]).rename(columns = {"percentage": "Smoking (%)"})

    df_case_copd_pct = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "LTC_COPD", metric)
    df_case_copd_pct = df_case_copd_pct[df_case_copd_pct["LTC_COPD"]==1].reset_index(drop=True).drop(columns = ["count","LTC_COPD"]).rename(columns = {"percentage": "COPD (%)"})

    dict_tables = {"Cancer cohort":df_case_pct,
                   "Control cohort":df_control_pct,
                   "Stage":df_case_stage_pct,
                   "Route":df_case_route_pct,
                   "Deprivation": df_case_deprivation_pct,
                   "Smoking": df_case_smoking_pct,
                   "COPD":df_case_copd_pct}
        
    for name,table in dict_tables.items():
        dict_tables[name] = table[table[metric]==1].drop(columns = [metric])

    df_row = pd.concat(dict_tables.values(), axis = 1)
    df_row["condition"] = metric
    df_mh_metrics = pd.concat([df_row,df_mh_metrics], axis = 0)

df_mh_metrics.sort_values("Cancer cohort (%)", ascending=False)

# COMMAND ----------

df_stage_by_mhi = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "tumour_stage_group", var_mh)


sns.barplot(hue="tumour_stage_group",
            y="percentage",
            x = var_mh,
            data= df_stage_by_mhi
)

plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

display(df_stage_by_mhi)

# COMMAND ----------

df_route_by_mhi = patient_pathways_utils.count_and_pct(df_pd_patient_flags, "route_earliest", var_mh)

display(df_route_by_mhi)

sns.barplot(hue="route_earliest",
            y="percentage",
            x = var_mh,
            data= df_route_by_mhi
)

plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

df = patient_pathways_utils.count_and_pct(df_pd_patient_flags,"imd_decile_group",  var_mh)
                                     
sns.barplot(hue="imd_decile_group",
            y="percentage",
            x = var_mh,
            data= patient_pathways_utils.count_and_pct(df_pd_patient_flags,"imd_decile_group",  var_mh)

)

plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

display(df)

# COMMAND ----------

df = patient_pathways_utils.count_and_pct(df_pd_patient_flags,"LTC_COPD",  var_mh)
                                     
sns.barplot(hue="LTC_COPD",
            y="percentage",
            x = var_mh,
            data= patient_pathways_utils.count_and_pct(df_pd_patient_flags,"LTC_COPD",  var_mh)

)

plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

display(df)

# COMMAND ----------

df = patient_pathways_utils.count_and_pct(df_pd_patient_flags,"Smoking_Flag",  var_mh)
                                     
sns.barplot(hue="Smoking_Flag",
            y="percentage",
            x = var_mh,
            data= patient_pathways_utils.count_and_pct(df_pd_patient_flags,"Smoking_Flag",  var_mh)

)

plt.ylabel("Percentage")
plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Emergency Presentation
# MAGIC

# COMMAND ----------

df_pd_patient_flags[df_pd_patient_flags["route_earliest"]=="Emergency presentation"].describe()

# COMMAND ----------

df_pd_patient_flags[df_pd_patient_flags["route_earliest"]!="Emergency presentation"].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # Activity count

# COMMAND ----------

stop_lin_reg_days_before = history_days/2 #when to stop the linear regression fit to activity (e.g. if this was 100, and history_days was 365, uses the first 265 days to fit the straight line)

# COMMAND ----------

# cross join for base dataframe with days, dataset, ID, count

df_base_table = patient_pathways_utils.create_base_table_from_lists(list1=unique_days, 
                                                                    name1="days_between_activity_diagnosis",
                                                                    list2=list_unique_datasets,
                                                                    name2="dataset",
                                                                    list3=unique_ids,
                                                                    name3="Patient_ID")

# sum of events per patient, per day, per dataset 

df_patient_activity_before_diagnosis = df_all_activity_before_diagnosis.filter((F.col("days_between_activity_diagnosis")<=history_days) 
                                                                               ).groupby(["days_between_activity_diagnosis", "dataset", "Patient_ID"]).count().toPandas()


df_patient_activity_before_diagnosis = df_base_table.merge(df_patient_activity_before_diagnosis, on = ["days_between_activity_diagnosis", "dataset", "Patient_ID"], how="left").fillna(0)

df_patient_activity_before_diagnosis_cumsum = df_patient_activity_before_diagnosis.sort_values(["dataset", "Patient_ID","days_between_activity_diagnosis"], ascending = False)
df_patient_activity_before_diagnosis_cumsum["cumsum_count"] = df_patient_activity_before_diagnosis_cumsum.groupby(["dataset", "Patient_ID"])["count"].cumsum()

df_patient_activity_before_diagnosis_cumsum = df_patient_activity_before_diagnosis_cumsum.merge(df_pd_patient_flags[["Patient_ID", "Sex" ,"tumour_stage_group", "imd_decile_group", "route_earliest", var_mh] + list_of_mh_fields], on = "Patient_ID", how="left")

for dataset in list_unique_datasets:
    # number of unique cases
    unique_ids_with_activity = df_all_activity_before_diagnosis.filter((F.col("days_between_activity_diagnosis")<=history_days) & (F.col("dataset")==dataset) ).select("Patient_ID").distinct().count()
    # total number
    total_events_of_activity = df_all_activity_before_diagnosis.filter((F.col("days_between_activity_diagnosis")<=history_days) & (F.col("dataset")==dataset) ).count()
    print(dataset)
    print("unique ids with activity", unique_ids_with_activity)
    print("total events of activity", total_events_of_activity)
    print("\n")

# COMMAND ----------

# cumulative count of activity for control

# cross join for base dataframe with days, dataset, ID, count
unique_ids_control = df_pd_patient_flags_control["Patient_ID"].unique()
df_base_table_control = patient_pathways_utils.create_base_table_from_lists(list1=unique_days, 
                                                                            name1="days_between_activity_diagnosis",
                                                                            list2=list_unique_datasets,
                                                                            name2="dataset",
                                                                            list3=unique_ids_control,
                                                                            name3="Patient_ID")

# sum of events per patient, per day, per dataset 

df_patient_activity_before_diagnosis_control = df_all_activity_before_diagnosis_control.filter((F.col("days_between_activity_diagnosis")<=history_days) 
                                                                               ).groupby(["days_between_activity_diagnosis", "dataset", "Patient_ID"]).count().toPandas()


df_patient_activity_before_diagnosis_control = df_base_table_control.merge(df_patient_activity_before_diagnosis_control, on = ["days_between_activity_diagnosis", "dataset", "Patient_ID"], how="left").fillna(0)

df_patient_activity_before_diagnosis_cumsum_control = df_patient_activity_before_diagnosis_control.sort_values(["dataset", "Patient_ID","days_between_activity_diagnosis"], ascending = False)
df_patient_activity_before_diagnosis_cumsum_control["cumsum_count"] = df_patient_activity_before_diagnosis_cumsum_control.groupby(["dataset", "Patient_ID"])["count"].cumsum()

df_patient_activity_before_diagnosis_cumsum_control = df_patient_activity_before_diagnosis_cumsum_control.merge(df_pd_patient_flags_control[["Patient_ID", "Sex","tumour_stage_group", "imd_decile_group", "route_earliest", var_mh] + list_of_mh_fields], on = "Patient_ID", how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Days to First GP Appointment

# COMMAND ----------

routes = df_patient_flags.select("Patient_ID", "route_earliest").distinct().toPandas()
df_patient_activity_before_diagnosis = df_patient_activity_before_diagnosis.merge(routes, on="Patient_ID", how="left")

# COMMAND ----------

df_gp_activity = df_patient_activity_before_diagnosis[(df_patient_activity_before_diagnosis["dataset"] == "GP_Appointments") & (df_patient_activity_before_diagnosis["count"] > 0)]
df_first_gp_app = df_gp_activity.loc[df_gp_activity.groupby("Patient_ID")["days_between_activity_diagnosis"].idxmax()]
fig, ax = plt.subplots(1,1, figsize=(10,5))

for route, color in pal_route.items():
    cumsum = {}
    route_df = df_first_gp_app[df_first_gp_app["route_earliest"] == route]
    total_route_patients = len(route_df)
    for i in range(365):
        days_before_diagnosis = 365-i
        val = len(route_df[route_df["days_between_activity_diagnosis"] >= days_before_diagnosis])
        cumsum[i] = 100*val/total_route_patients
    
    ax.plot(cumsum.keys(), cumsum.values(), label=route, color=color)
ax.set_xlabel("days before diagnosis")
ax.set_ylabel("% of patients")
ax.set_title("% of Patients with a GP App. Prior to Diagnosis")
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Average activity over time

# COMMAND ----------

df_patient_activity_before_diagnosis_cumsum["week"] = df_patient_activity_before_diagnosis_cumsum["days_between_activity_diagnosis"].apply(lambda x: int(x/7))
df_activity_by_week = df_patient_activity_before_diagnosis_cumsum.groupby(["Patient_ID", "dataset", "Sex" ,"tumour_stage_group", "imd_decile_group", "route_earliest", "week"])["count"].sum().to_frame().reset_index()

df_patient_activity_before_diagnosis_cumsum_control["week"] = df_patient_activity_before_diagnosis_cumsum_control["days_between_activity_diagnosis"].apply(lambda x: int(x/7))
df_activity_by_week_control = df_patient_activity_before_diagnosis_cumsum_control.groupby(["Patient_ID", "dataset","Sex" , "tumour_stage_group", "imd_decile_group", "route_earliest", "week"])["count"].sum().to_frame().reset_index()
                                                                                                                                                                                                                 

# COMMAND ----------

sns.set(font_scale = 1.25, style="whitegrid")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(15,6))

        df = df_activity_by_week[(df_activity_by_week["dataset"]==dataset)]
        df_control = df_activity_by_week_control[(df_activity_by_week_control["dataset"]==dataset)]

        xlim = [0,int(history_days/7)-1]
        ax.set_xlim(xlim)


        sns.lineplot(x="week",
                    y="count",
                    data=df, 
                    ax = ax,
                    label="Lung cancer cases")

        sns.lineplot(x="week",
                    y="count",
                    data=df_control, 
                    ax = ax,
                    label="Controls")
        
        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Weeks between activity and diagnosis")
        plt.ylabel("Average number of weekly events per patient")

# COMMAND ----------

sns.set(font_scale = 1.25, style="whitegrid")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(15,6))
        
        df = df_activity_by_week[(df_activity_by_week["dataset"]==dataset) & (df_activity_by_week["tumour_stage_group"].isin(["early", "late"]))]

        xlim = [0,int(history_days/7)-1]
        ax.set_xlim(xlim)


        sns.lineplot(x="week",
                    y="count",
                    data=df,
                    hue = "tumour_stage_group",
                    hue_order = ["early", "late"],
                    palette=pal_stage,
                    ax = ax)

        
        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Weeks between activity and diagnosis")
        plt.ylabel("Average number of weekly events per patient")

# COMMAND ----------

sns.set(font_scale = 1.25, style="whitegrid")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(15,6))

        df = df_activity_by_week[(df_activity_by_week["dataset"]==dataset)]
        df = df[df["imd_decile_group"].isin(["decile_1_to_3", "decile_4_to_7", "decile_8_to_10"])]
        
        xlim = [0,int(history_days/7)-1]
        ax.set_xlim(xlim)

        sns.lineplot(x="week",
                    y="count",
                    data=df,
                    hue = "imd_decile_group" ,
                    ax = ax,
                    palette=pal_imd)

        
        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Weeks between activity and diagnosis")
        plt.ylabel("Average number of weekly events per patient")



# COMMAND ----------

sns.set(font_scale = 1.25, style="whitegrid")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(15,6))
        
        df = df_activity_by_week[(df_activity_by_week["dataset"]==dataset)]
        df = df[df["route_earliest"].isin(["GP referral", "USC", "Emergency presentation", "Other outpatient"])]
        
        xlim = [0,int(history_days/7)-1]
        ax.set_xlim(xlim)

        sns.lineplot(x="week",
                    y="count",
                    data=df,
                    hue = "route_earliest" ,
                    ax = ax)

        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Weeks between activity and diagnosis")
        plt.ylabel("Average number of weekly events per patient")



# COMMAND ----------

sns.set(font_scale = 1.25, style="whitegrid")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":

        df = df_activity_by_week[(df_activity_by_week["dataset"]==dataset)]
        
        fig, ax = plt.subplots(figsize=(15,6))       

        xlim = [0,int(history_days/7)-1]
        ax.set_xlim(xlim)

        sns.lineplot(x="week",
                    y="count",
                    data=df,
                    hue = "Sex",
                    hue_order = ["F", "M"],
                    ax = ax)

       
        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Weeks between activity and diagnosis")
        plt.ylabel("Average number of weekly events per patient")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cumulative

# COMMAND ----------

sns.set(font_scale = 1.25, style="white")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(15,6))       

        df = df_patient_activity_before_diagnosis_cumsum[(df_patient_activity_before_diagnosis_cumsum["dataset"]==dataset)]
        df_control = df_patient_activity_before_diagnosis_cumsum_control[(df_patient_activity_before_diagnosis_cumsum_control["dataset"]==dataset)]

        xlim = [0,history_days]
        ax.set_xlim(xlim)

        df_mean_per_day = df.groupby("days_between_activity_diagnosis")["cumsum_count"].mean().reset_index()

        sns.regplot(data=df_mean_per_day[df_mean_per_day["days_between_activity_diagnosis"]>=stop_lin_reg_days_before],
                    x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    scatter=False,
                    ax= ax,
                    truncate = False,
                    label = f"Cases straight line fit to first {history_days-stop_lin_reg_days_before} days")
        
        sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    data=df, 
                    ax = ax,
                    label="Lung cancer cases")

        sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    data=df_control, 
                    ax = ax,
                    label="Controls")
        
        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Days between activity and diagnosis")
        plt.ylabel("Average cumulative number of events per patient")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### by stage

# COMMAND ----------

sns.set(font_scale = 1.2, style="whitegrid")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(16,6))

        df = df_patient_activity_before_diagnosis_cumsum[(df_patient_activity_before_diagnosis_cumsum["dataset"]==dataset)]
        df = df[df["tumour_stage_group"].isin(["early", "late"])]

        xlim = [0,history_days]
        ax.set_xlim(xlim)

        df_mean_per_day_early = df[df["tumour_stage_group"]=="early"].groupby("days_between_activity_diagnosis")["cumsum_count"].mean().reset_index()
        df_mean_per_day_late = df[df["tumour_stage_group"]=="late"].groupby("days_between_activity_diagnosis")["cumsum_count"].mean().reset_index()

        sns.regplot(data=df_mean_per_day_early[df_mean_per_day_early["days_between_activity_diagnosis"]>=stop_lin_reg_days_before],
                    x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    scatter=False,
                    ax= ax,
                    truncate = False,
                    color='yellowgreen',
                    label = f"Straight line fit to early stage")

        sns.regplot(data=df_mean_per_day_late[df_mean_per_day_late["days_between_activity_diagnosis"]>=stop_lin_reg_days_before],
                    x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    scatter=False,
                    ax= ax,
                    truncate = False,
                    color="salmon",
                    label = f"Straight line fit to late stage")
        

        sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    hue="tumour_stage_group",
                    data=df, 
                    ax = ax,
                    palette = pal_stage)

        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Days between activity and diagnosis")
        plt.ylabel("Average cumulative number of events per patient")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### imd decile group

# COMMAND ----------

sns.set(font_scale = 1.2, style="white")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(16,6))       

        df = df_patient_activity_before_diagnosis_cumsum[(df_patient_activity_before_diagnosis_cumsum["dataset"]==dataset)]
        df = df[df["imd_decile_group"].isin(["decile_1_to_3", "decile_4_to_7", "decile_8_to_10"])]

        xlim = [0,history_days]
        ax.set_xlim(xlim)

        sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    hue="imd_decile_group",
                    data=df, 
                    ax = ax,
                    palette = pal_imd)

        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Days between activity and diagnosis")
        plt.ylabel("Average cumulative number of events per patient")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### route of diagnosis

# COMMAND ----------

sns.set(font_scale = 1.2, style="white")
for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(16,6))

        df = df_patient_activity_before_diagnosis_cumsum[(df_patient_activity_before_diagnosis_cumsum["dataset"]==dataset)]
        df = df[df["route_earliest"].isin(["GP referral", "USC", "Emergency presentation", "Unknown"])]
        
        xlim = [0,history_days]
        ax.set_xlim(xlim)

        sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    hue="route_earliest",
                    data=df,
                    palette=pal_route, 
                    ax = ax,)

        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Days between activity and diagnosis")
        plt.ylabel("Average cumulative number of events per patient")

# COMMAND ----------

sns.set(font_scale = 1.2, style="white")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(14,4))

        df = df_patient_activity_before_diagnosis_cumsum[(df_patient_activity_before_diagnosis_cumsum["dataset"]==dataset)]
        df = df[df["route_earliest"].isin(["GP referral", "Emergency presentation"])]

        xlim = [0,history_days]
        ax.set_xlim(xlim)

        sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    hue="route_earliest",
                    data=df, 
                    ax = ax,)

        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Days between activity and diagnosis")
        plt.ylabel("Average cumulative number of events per patient")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Sex

# COMMAND ----------

sns.set(font_scale = 1.2, style="white")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(16,6))

        df = df_patient_activity_before_diagnosis_cumsum[(df_patient_activity_before_diagnosis_cumsum["dataset"]==dataset)]

        xlim = [0,history_days]
        ax.set_xlim(xlim)


        sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    hue="Sex",
                    hue_order = ["F", "M"],
                    data=df, 
                    ax = ax,)

        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Days between activity and diagnosis")
        plt.ylabel("Average cumulative number of events per patient")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### SMI

# COMMAND ----------

sns.set(font_scale = 1.2, style="white")

for dataset in list_unique_datasets:

    if dataset != "Cancer_registry":
        fig, ax = plt.subplots(figsize=(16,6))

        df = df_patient_activity_before_diagnosis_cumsum[(df_patient_activity_before_diagnosis_cumsum["dataset"]==dataset)]

        xlim = [0,history_days]
        ax.set_xlim(xlim)


        sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumsum_count",
                    hue=var_mh,
                    data=df, 
                    ax = ax,)

        plt.xticks(rotation=90);
        plt.title(dataset)
        plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
        ax.invert_xaxis()
        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.xlabel("Days between activity and diagnosis")
        plt.ylabel("Average cumulative number of events per patient")

# COMMAND ----------

# MAGIC %md
# MAGIC ## investigate codes which have increased over time

# COMMAND ----------

# MAGIC %md
# MAGIC ### GP events

# COMMAND ----------

df_number_unique_patients_per_code_comparison_cases_gp = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis,
                                                                                                       dataset="gp_events",
                                                                                                       total_num_unique_patients = df_pd_patient_flags.shape[0],
                                                                                                       column_for_activity = "description",
                                                                                                       history_days=history_days).toPandas()

display(df_number_unique_patients_per_code_comparison_cases_gp)

# COMMAND ----------

df = df_number_unique_patients_per_code_comparison_cases_gp[(df_number_unique_patients_per_code_comparison_cases_gp["percentage_of_cases_second_half"]>=10) & (df_number_unique_patients_per_code_comparison_cases_gp["ratio_second_half_to_first_half"]>=1.5)]

sns.set(font_scale = 1.2, style="white")
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x="percentage_of_cases_first_half", y="percentage_of_cases_second_half", hue = "description", data = df, style = "description", markers= True, s=75)

plt.xlabel("Percentage of patients in earlier period (%)")
plt.ylabel("Percentage of patients in later period (%)")
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = "GP")

# COMMAND ----------

df_number_unique_patients_per_code_comparison_controls = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis_control,
                                                                                                          dataset="gp_events",
                                                                                                          total_num_unique_patients = df_pd_patient_flags.shape[0],
                                                                                                          column_for_activity = "description",
                                                                                                          history_days=history_days)

display(df_number_unique_patients_per_code_comparison_controls)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ECDS

# COMMAND ----------

df_number_unique_patients_per_code_comparison_cases_ecds = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis,
                                                                                dataset="ECDS",
                                                                                total_num_unique_patients = df_pd_patient_flags.shape[0],
                                                                                column_for_activity = "Emergency_Care_Chief_Complaint",
                                                                                history_days=history_days).toPandas()

df_number_unique_patients_per_code_comparison_cases_ecds

# COMMAND ----------

df = df_number_unique_patients_per_code_comparison_cases_ecds[df_number_unique_patients_per_code_comparison_cases_ecds["percentage_of_cases_second_half"]>=2]

sns.set(font_scale = 1.2, style="white")
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x="percentage_of_cases_first_half", y="percentage_of_cases_second_half", hue = "Emergency_Care_Chief_Complaint", data = df, style = "Emergency_Care_Chief_Complaint", markers= True, s=75)

plt.xlabel("Percentage of patients in earlier period (%)")
plt.ylabel("Percentage of patients in later period (%)")
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = "Chief Complaint")

# COMMAND ----------

df_number_unique_patients_per_code_comparison_controls_ecds = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis_control,
                                                                                dataset="ECDS",
                                                                                total_num_unique_patients = df_pd_patient_flags_control.shape[0],
                                                                                column_for_activity = "Emergency_Care_Chief_Complaint",
                                                                                history_days=history_days)

display(df_number_unique_patients_per_code_comparison_controls_ecds)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Acute

# COMMAND ----------

df_number_unique_patients_per_code_comparison_cases_acute = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis,
                                                                                dataset="Acute_All",
                                                                                total_num_unique_patients = df_pd_patient_flags.shape[0],
                                                                                column_for_activity = "ICD10_description",
                                                                                history_days=history_days).toPandas()

display(df_number_unique_patients_per_code_comparison_cases_acute)

# COMMAND ----------

df = df_number_unique_patients_per_code_comparison_cases_acute[df_number_unique_patients_per_code_comparison_cases_acute["percentage_of_cases_second_half"]>=1.5]

sns.set(font_scale = 1.2, style="white")
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x="percentage_of_cases_first_half", y="percentage_of_cases_second_half", hue = "ICD10_description", data = df, style = "ICD10_description", markers = True, s = 75)

plt.xlabel("Percentage of patients in earlier period (%)")
plt.ylabel("Percentage of patients in later period (%)")
plt.xlim(left = -0.1)
plt.ylim(bottom = -0.1)
plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = "Diagnoses")

# COMMAND ----------

df_number_unique_patients_per_code_comparison_controls_acute = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis_control,
                                                                                dataset="Acute_All",
                                                                                total_num_unique_patients = df_pd_patient_flags_control.shape[0],
                                                                                column_for_activity = "ICD10_description",
                                                                                history_days=history_days)

display(df_number_unique_patients_per_code_comparison_controls_acute)

# COMMAND ----------

display(df_all_activity_before_diagnosis.filter(F.col("dataset")=="Acute_All"))

# COMMAND ----------

# Procedures
df_number_unique_patients_per_code_comparison_cases_acute = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis,
                                                                                dataset="Acute_All",
                                                                                total_num_unique_patients = df_pd_patient_flags.shape[0],
                                                                                column_for_activity = "Title",
                                                                                history_days=history_days).toPandas()

df = df_number_unique_patients_per_code_comparison_cases_acute[df_number_unique_patients_per_code_comparison_cases_acute["percentage_of_cases_second_half"]>=1.5]

sns.set(font_scale = 1.2, style="white")
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x="percentage_of_cases_first_half", y="percentage_of_cases_second_half", hue = "Title", data = df, style = "Title", markers = True)

plt.xlabel("Percentage of patients in earlier period (%)")
plt.ylabel("Percentage of patients in later period (%)")
plt.xlim(left = -0.1)
plt.ylim(bottom = -0.1)
plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = "Chief Complaint")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 111 calls

# COMMAND ----------

df_number_unique_patients_per_code_comparison_cases_111 = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis,
                                                                                dataset="UC_111_All",
                                                                                total_num_unique_patients = df_pd_patient_flags.shape[0],
                                                                                column_for_activity = "SG_Description",
                                                                                history_days=history_days).toPandas()

display(df_number_unique_patients_per_code_comparison_cases_111)

# COMMAND ----------

df = df_number_unique_patients_per_code_comparison_cases_111[df_number_unique_patients_per_code_comparison_cases_111["percentage_of_cases_second_half"]>=0.75]

sns.set(font_scale = 1.2, style="white")
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x="percentage_of_cases_first_half", y="percentage_of_cases_second_half", hue = "SG_Description", data = df, style = "SG_Description", s = 75)

plt.xlabel("Percentage of patients in earlier period (%)")
plt.ylabel("Percentage of patients in later period (%)")
plt.xlim(left = -0.1)
plt.ylim(bottom = -0.1)
plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = "111 Chief Complaint")

# COMMAND ----------

df = df_number_unique_patients_per_code_comparison_cases_111[df_number_unique_patients_per_code_comparison_cases_111["percentage_of_cases_second_half"]>=0.75]
df=df[~df["SG_Description"].isin(non_symptoms_111)]

sns.set(font_scale = 1.2, style="white")
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x="percentage_of_cases_first_half", y="percentage_of_cases_second_half", hue = "SG_Description", data = df, style = "SG_Description", s = 75, alpha = 0.75)

plt.xlabel("Percentage of patients in earlier period (%)")
plt.ylabel("Percentage of patients in later period (%)")
plt.xlim(left = -0.1)
plt.ylim(bottom = -0.1)
plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = "111 Chief Complaint")

# COMMAND ----------

df_number_unique_patients_per_code_comparison_controls_111 = patient_pathways_utils.compare_codes_in_activity(df_all_activity_before_diagnosis_control,
                                                                                dataset="UC_111_All",
                                                                                total_num_unique_patients = df_pd_patient_flags.shape[0],
                                                                                column_for_activity = "SG_Description",
                                                                                history_days=history_days)

display(df_number_unique_patients_per_code_comparison_controls_111)

# COMMAND ----------

# top symptoms from 111
n = 12

df_all_activity_before_diagnosis.filter((F.col("days_between_activity_diagnosis")<=history_days) &
                                        (F.col("dataset")=="UC_111_All")).groupby("SG_Description").count().display()

df_111_activity_before_diagnosis = df_all_activity_before_diagnosis.filter((F.col("days_between_activity_diagnosis")<=history_days) &
                                                                           (F.col("dataset")=="UC_111_All"))

df_111_activity_before_diagnosis_count = df_111_activity_before_diagnosis.groupby(["days_between_activity_diagnosis", "SG_Description"]).count().toPandas()

df_111_activity_before_diagnosis_cumsum = df_111_activity_before_diagnosis_count.sort_values(["SG_Description", "days_between_activity_diagnosis"], ascending = False)
df_111_activity_before_diagnosis_cumsum["cumsum_count"] = df_111_activity_before_diagnosis_cumsum.groupby("SG_Description")["count"].cumsum()

top_n_111_symptoms = list(df_111_activity_before_diagnosis_cumsum.groupby("SG_Description")["cumsum_count"].max().sort_values(ascending= False).to_frame().reset_index()["SG_Description"].iloc[0:n])

fig, ax = plt.subplots(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.lineplot(x="days_between_activity_diagnosis",
            y="cumsum_count",
            hue="SG_Description",
            data = df_111_activity_before_diagnosis_cumsum[df_111_activity_before_diagnosis_cumsum["SG_Description"].isin(top_n_111_symptoms)])

ax.invert_xaxis()
plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
plt.ylabel("Cumulative count")

plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = "SG_Description")



# COMMAND ----------

symptoms_only = list(set(top_n_111_symptoms) - set(non_symptoms_111))

fig, ax = plt.subplots(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.lineplot(x="days_between_activity_diagnosis",
            y="cumsum_count",
            hue="SG_Description",
            data = df_111_activity_before_diagnosis_cumsum[df_111_activity_before_diagnosis_cumsum["SG_Description"].isin(symptoms_only)],
            hue_order = [symptom for symptom in top_n_111_symptoms if symptom in symptoms_only])

ax.invert_xaxis()
plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
plt.ylabel("Cumulative count")

plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = "SG_Description")

# COMMAND ----------

# cross join for base dataframe with days, SG_Description, ID, count

df_base_table = patient_pathways_utils.create_base_table_from_lists(list1=unique_days, 
                                                                    name1="days_between_activity_diagnosis",
                                                                    list2=top_n_111_symptoms,
                                                                    name2="SG_Description",
                                                                    list3=unique_ids,
                                                                    name3="Patient_ID")
# sum of events per patient, per day, per symptom 

df_patient_activity_before_diagnosis = df_111_activity_before_diagnosis.filter((F.col("days_between_activity_diagnosis")<=history_days) 
                                                                               ).groupby(["days_between_activity_diagnosis", "SG_Description", "Patient_ID"]).count().toPandas()


df_patient_activity_before_diagnosis = df_base_table.merge(df_patient_activity_before_diagnosis, on = ["days_between_activity_diagnosis", "SG_Description", "Patient_ID"], how="left").fillna(0)

df_patient_activity_before_diagnosis_cumsum = df_patient_activity_before_diagnosis.sort_values(["SG_Description", "Patient_ID","days_between_activity_diagnosis"], ascending = False)
df_patient_activity_before_diagnosis_cumsum["cumsum_count"] = df_patient_activity_before_diagnosis_cumsum.groupby(["SG_Description", "Patient_ID"])["count"].cumsum()

df_patient_activity_before_diagnosis_cumsum = df_patient_activity_before_diagnosis_cumsum.merge(df_pd_patient_flags[["Patient_ID", "tumour_stage_group", "imd_decile_group", "route_earliest"]], on = "Patient_ID", how="left")

df_patient_activity_before_diagnosis_cumsum["month"] = df_patient_activity_before_diagnosis_cumsum["days_between_activity_diagnosis"].apply(lambda x: int(x/(365/12)))

df_activity_by_period_111_symptom = df_patient_activity_before_diagnosis_cumsum.groupby(["Patient_ID", "SG_Description", "tumour_stage_group", "imd_decile_group", "route_earliest", "month"])["count"].sum().to_frame().reset_index()


# COMMAND ----------

# cross join for base dataframe with days, SG_Description, ID, count

df_base_table_control = patient_pathways_utils.create_base_table_from_lists(list1=unique_days, 
                                                                            name1="days_between_activity_diagnosis",
                                                                            list2=top_n_111_symptoms,
                                                                            name2="SG_Description",
                                                                            list3=unique_ids_control,
                                                                            name3="Patient_ID")

df_111_activity_before_diagnosis_control = df_all_activity_before_diagnosis_control.filter((F.col("days_between_activity_diagnosis")<=history_days) &
                                                                           (F.col("dataset")=="UC_111_All"))

df_patient_activity_before_diagnosis_control = df_111_activity_before_diagnosis_control.filter((F.col("days_between_activity_diagnosis")<=history_days) 
                                                                               ).groupby(["days_between_activity_diagnosis", "SG_Description", "Patient_ID"]).count().toPandas()


df_patient_activity_before_diagnosis_control = df_base_table_control.merge(df_patient_activity_before_diagnosis_control, on = ["days_between_activity_diagnosis", "SG_Description", "Patient_ID"], how="left").fillna(0)

df_patient_activity_before_diagnosis_control_cumsum = df_patient_activity_before_diagnosis_control.sort_values(["SG_Description", "Patient_ID","days_between_activity_diagnosis"], ascending = False)
df_patient_activity_before_diagnosis_control_cumsum["cumsum_count"] = df_patient_activity_before_diagnosis_control_cumsum.groupby(["SG_Description", "Patient_ID"])["count"].cumsum()

df_patient_activity_before_diagnosis_control_cumsum = df_patient_activity_before_diagnosis_control_cumsum.merge(df_pd_patient_flags_control[["Patient_ID", "tumour_stage_group", "imd_decile_group", "route_earliest"]], on = "Patient_ID", how="left")

df_patient_activity_before_diagnosis_control_cumsum["month"] = df_patient_activity_before_diagnosis_control_cumsum["days_between_activity_diagnosis"].apply(lambda x: int(x/(365/12)))

df_activity_by_period_111_symptom_control = df_patient_activity_before_diagnosis_control_cumsum.groupby(["Patient_ID", "SG_Description", "tumour_stage_group", "imd_decile_group", "route_earliest", "month"])["count"].sum().to_frame().reset_index()

# COMMAND ----------

for symptom in top_n_111_symptoms:

    fig, ax = plt.subplots(figsize=(15,6))
    sns.set(font_scale = 1.25, style="whitegrid")

    df = df_activity_by_period_111_symptom[(df_activity_by_period_111_symptom["SG_Description"]==symptom)]
    df_control = df_activity_by_period_111_symptom_control[(df_activity_by_period_111_symptom_control["SG_Description"]==symptom)]

    xlim = [0,12]
    ax.set_xlim(xlim)

    sns.lineplot(x="month",
                y="count",
                data=df, 
                ax = ax,
                label="Lung cancer cases")

    sns.lineplot(x="month",
                y="count",
                data=df_control, 
                ax = ax,
                label="Controls")
    
    plt.xticks(rotation=90);
    plt.title(symptom)
    plt.legend(bbox_to_anchor=[1.05,1], loc=2, title = dataset)
    ax.invert_xaxis()
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.xlabel("Months between activity and diagnosis")
    plt.ylabel("Average number of calls per patient")

# COMMAND ----------

# MAGIC %md
# MAGIC # NG12 and symptoms events 

# COMMAND ----------

df_dates_symptoms_ng12_count = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags,
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)

df_dates_symptoms_ng12_count["population"] = "Cases"

df_dates_symptoms_ng12_control_count = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags_control,
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)

df_dates_symptoms_ng12_control_count["population"] = "Controls"

df_dates_symptoms_ng12_cases_and_controls = pd.concat([df_dates_symptoms_ng12_count, df_dates_symptoms_ng12_control_count], axis = 0)

df_dates_symptoms_ng12_cases_and_controls["symptom"] = df_dates_symptoms_ng12_cases_and_controls["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_dates_symptoms_ng12_cases_and_controls["dataset"] = df_dates_symptoms_ng12_cases_and_controls["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))


df_dates_symptoms_ng12_cases_and_controls


# COMMAND ----------

plt.figure(figsize=(10,8))
sns.set(font_scale = 1, style="whitegrid")

df = df_dates_symptoms_ng12_cases_and_controls[df_dates_symptoms_ng12_cases_and_controls["population"]=="Cases"]

ax = sns.lineplot(x="days_between_activity_diagnosis",
                  y="cumulative_percentage",
                  data = df[df["col_name"].isin(date_columns_ng12)],
                  hue="col_name")

plt.xticks(rotation=90)
ax.invert_xaxis()

plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

plt.xlabel("Days before cancer diagnosis")
plt.ylabel("Cumulative percentage of population")

plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

sns.set(font_scale = 1, style="whitegrid")

for date_col in date_columns_ng12:

    plt.figure(figsize=(10,8))

    ax = sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumulative_percentage",
                    data = df_dates_symptoms_ng12_cases_and_controls[df_dates_symptoms_ng12_cases_and_controls["col_name"]== date_col],
                    hue = "population" )
    
    plt.xticks(rotation=90)
    ax.invert_xaxis()

    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.title(date_col)
    plt.xlabel("Days before cancer diagnosis")
    plt.ylabel("Cumulative percentage of population")

    plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### breakdown by age

# COMMAND ----------

df_dates_symptoms_ng12_count_younger = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[df_pd_patient_flags["Age"]<=74],
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "Age", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)

df_dates_symptoms_ng12_count_younger["population"] = "<=74"

df_dates_symptoms_ng12_count_older = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[df_pd_patient_flags["Age"]>74],
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "Age", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)

df_dates_symptoms_ng12_count_older["population"] = ">74"

df_dates_symptoms_ng12_younger_older = pd.concat([df_dates_symptoms_ng12_count_younger, df_dates_symptoms_ng12_count_older], axis = 0)

df_dates_symptoms_ng12_younger_older["symptom"] = df_dates_symptoms_ng12_younger_older["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_dates_symptoms_ng12_younger_older["dataset"] = df_dates_symptoms_ng12_younger_older["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

# COMMAND ----------

for col in date_columns_ng12:
    plt.figure(figsize=(10,8))
    sns.set(font_scale = 1, style="whitegrid")

    df = df_dates_symptoms_ng12_younger_older[df_dates_symptoms_ng12_younger_older["col_name"]==col]

    ax = sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumulative_percentage",
                    data = df,
                    hue="population")

    plt.xticks(rotation=90)
    ax.invert_xaxis()

    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.title(col)
    plt.xlabel("Days before cancer diagnosis")
    plt.ylabel("Cumulative percentage of population")

    plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### breakdown by stage

# COMMAND ----------

df_dates_symptoms_ng12_count_early = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[df_pd_patient_flags["tumour_stage_group"]=="early"],
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)

df_dates_symptoms_ng12_count_early["population"] = "early"

df_dates_symptoms_ng12_count_late = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[df_pd_patient_flags["tumour_stage_group"]=="late"],
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)

df_dates_symptoms_ng12_count_late["population"] = "late"

df_dates_symptoms_ng12_early_late = pd.concat([df_dates_symptoms_ng12_count_early, df_dates_symptoms_ng12_count_late], axis = 0)

df_dates_symptoms_ng12_early_late["symptom"] = df_dates_symptoms_ng12_early_late["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_dates_symptoms_ng12_early_late["dataset"] = df_dates_symptoms_ng12_early_late["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))


# COMMAND ----------

# EARLY

plt.figure(figsize=(10,8))
sns.set(font_scale = 1, style="whitegrid")

df = df_dates_symptoms_ng12_early_late[df_dates_symptoms_ng12_early_late["population"]=="early"]

ax = sns.lineplot(x="days_between_activity_diagnosis",
                  y="cumulative_percentage",
                  data = df[df["col_name"].isin(date_columns_ng12)],
                  hue="col_name")

plt.xticks(rotation=90)
ax.invert_xaxis()

plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

plt.xlabel("Days before cancer diagnosis")
plt.ylabel("Cumulative percentage of population")

plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

# LATE
plt.figure(figsize=(10,8))
sns.set(font_scale = 1, style="whitegrid")

df = df_dates_symptoms_ng12_early_late[df_dates_symptoms_ng12_early_late["population"]=="late"]

ax = sns.lineplot(x="days_between_activity_diagnosis",
                  y="cumulative_percentage",
                  data = df[df["col_name"].isin(date_columns_ng12)],
                  hue="col_name")

plt.xticks(rotation=90)
ax.invert_xaxis()

plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

plt.xlabel("Days before cancer diagnosis")
plt.ylabel("Cumulative percentage of population")

plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### SMI

# COMMAND ----------

df_dates_symptoms_ng12_count_smi = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[df_pd_patient_flags[var_mh]==1],
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)

df_dates_symptoms_ng12_count_smi["population"] = "SMI"

df_dates_symptoms_ng12_count_no_smi = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[df_pd_patient_flags[var_mh]==0],
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)

df_dates_symptoms_ng12_count_no_smi["population"] = "No SMI"

df_dates_symptoms_ng12_smi = pd.concat([df_dates_symptoms_ng12_count_smi, df_dates_symptoms_ng12_count_no_smi], axis = 0)

df_dates_symptoms_ng12_smi["symptom"] = df_dates_symptoms_ng12_smi["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_dates_symptoms_ng12_smi["dataset"] = df_dates_symptoms_ng12_smi["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

# COMMAND ----------

for col in date_columns_ng12:
    plt.figure(figsize=(10,8))   

    df = df_dates_symptoms_ng12_smi[df_dates_symptoms_ng12_smi["col_name"]== col]

    ax = sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumulative_percentage",
                    data = df,
                    hue = "population")

    plt.xticks(rotation=90)
    ax.invert_xaxis()

    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.title(col)
    plt.xlabel("Days before cancer diagnosis")
    plt.ylabel("Cumulative percentage of population")

    plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

sns.set(font_scale = 1.5, style="whitegrid")

for symptom in list_of_symptoms_incl_prescriptions:
    plt.figure(figsize=(10,8))   

    df = df_dates_symptoms_ng12_smi[df_dates_symptoms_ng12_smi["symptom"]== symptom]
    df = df[df["dataset"]=="overall"]

    ax = sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumulative_percentage",
                    data = df,
                    hue = "population")
    
    plt.xticks(rotation=90)
    ax.invert_xaxis()

    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.title(symptom)
    plt.xlabel("Days before cancer diagnosis")
    plt.ylabel("Cumulative percentage of population")

    plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### route

# COMMAND ----------

df_long_ng12_flags= pd.melt(df_pd_patient_flags, 
                            id_vars = ["Patient_ID", "route_earliest", "Sex", "tumour_stage_group", "imd_decile_group"],
                            value_vars = flag_columns_ng12)

# COMMAND ----------

# by route identify percentage which had different conditions met for referral
df_prop_ng12_by_route = df_long_ng12_flags.groupby(["route_earliest", "variable"])["value"].value_counts(normalize=True).to_frame()
df_prop_ng12_by_route.columns = ["Percentage"]
df_prop_ng12_by_route = df_prop_ng12_by_route.reset_index()
df_prop_ng12_by_route = df_prop_ng12_by_route[df_prop_ng12_by_route["value"]==1]
df_prop_ng12_by_route_wide = df_prop_ng12_by_route.pivot_table(index="route_earliest", columns = "variable", values = "Percentage")
df_prop_ng12_by_route_wide

# COMMAND ----------

df_dates_symptoms_ng12_route = pd.DataFrame()

for route in ["GP referral", "Emergency Presentation", "USC", "Unknown"]:
    df_route = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[df_pd_patient_flags["route_earliest"]==route],
                                                                  date_columns_to_process = date_columns_to_process,
                                                                  id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                  diagnosis_date_col = "diagnosis_date_earliest",
                                                                  history_days = history_days)
    
    df_route["population"] = route
    
    df_dates_symptoms_ng12_route = pd.concat([df_dates_symptoms_ng12_route, df_route], axis = 0)
    
df_dates_symptoms_ng12_route = df_dates_symptoms_ng12_route.reset_index(drop=True)
df_dates_symptoms_ng12_route["symptom"] = df_dates_symptoms_ng12_route["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_dates_symptoms_ng12_route["dataset"] = df_dates_symptoms_ng12_route["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

# COMMAND ----------

for col in date_columns_ng12:
    plt.figure(figsize=(10,8))   

    df = df_dates_symptoms_ng12_route[df_dates_symptoms_ng12_route["col_name"]== col]

    ax = sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumulative_percentage",
                    data = df,
                    hue = "population", 
                    hue_order = ["Emergency Presentation", "Unknown","GP referral", "USC"],
                    palette = pal_route)

    plt.xticks(rotation=90)
    ax.invert_xaxis()

    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.title(col)
    plt.xlabel("Days before cancer diagnosis")
    plt.ylabel("Cumulative percentage of population")

    plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

sns.set(font_scale = 1.5, style="whitegrid")

for symptom in list_of_symptoms_incl_prescriptions:
    plt.figure(figsize=(10,8))   

    df = df_dates_symptoms_ng12_route[df_dates_symptoms_ng12_route["symptom"]== symptom]
    df = df[df["dataset"]=="overall"]

    ax = sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumulative_percentage",
                    data = df,
                    hue = "population", 
                    hue_order = ["Emergency Presentation", "Unknown","GP referral", "USC"],
                    palette=pal_route)
    
    plt.xticks(rotation=90)
    ax.invert_xaxis()

    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.title(symptom)
    plt.xlabel("Days before cancer diagnosis")
    plt.ylabel("Cumulative percentage of population")

    plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Any Reported Symptom

# COMMAND ----------

# MAGIC %md
# MAGIC #### Individual symptoms

# COMMAND ----------

plt.figure(figsize=(10,8))
sns.set(font_scale = 1.5, style="whitegrid")

df = df_dates_symptoms_ng12_cases_and_controls[df_dates_symptoms_ng12_cases_and_controls["population"]=="Cases"]
df = df[df["dataset"]=="overall"]
df = df[df["population"]=="Cases"]
df = df[df["symptom"]!="NG12"]

ax = sns.lineplot(x="days_between_activity_diagnosis",
                  y="cumulative_percentage",
                  data = df,
                  hue="symptom",
                  hue_order = list(df[df["days_between_activity_diagnosis"]==0].sort_values(by="cumulative_percentage", ascending=False)["symptom"]),
                  palette = pal_symptom,
                  style = "symptom"
)

plt.xticks(rotation=90)
ax.invert_xaxis()

plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

plt.xlabel("Days before cancer diagnosis")
plt.ylabel("Cumulative percentage of population")

plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

sns.set(font_scale = 1.5, style="whitegrid")

for symptom in list_of_symptoms_incl_prescriptions:
    plt.figure(figsize=(10,8))   

    df = df_dates_symptoms_ng12_cases_and_controls[df_dates_symptoms_ng12_cases_and_controls["symptom"]== symptom]
    df = df[df["population"]=="Cases"]
    df = df[df["dataset"]!="overall"]

    ax = sns.lineplot(x="days_between_activity_diagnosis",
                    y="cumulative_percentage",
                    data = df,
                    hue = "dataset" )
    
    plt.xticks(rotation=90)
    ax.invert_xaxis()

    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.title(symptom)
    plt.xlabel("Days before cancer diagnosis")
    plt.ylabel("Cumulative percentage of population")

    plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

for date_col in date_columns_to_process:

    for symptom in list_of_symptoms_incl_prescriptions:
        if symptom in date_col:
            plt.figure(figsize=(10,8))
            sns.set(font_scale = 1.5, style="whitegrid")

            df = df_dates_symptoms_ng12_cases_and_controls[df_dates_symptoms_ng12_cases_and_controls["col_name"]== date_col]

            ax = sns.lineplot(x="days_between_activity_diagnosis",
                            y="cumulative_percentage",
                            data = df,
                            hue = "population" )
            
            plt.xticks(rotation=90)
            ax.invert_xaxis()

            plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

            plt.title(date_col)
            plt.xlabel("Days before cancer diagnosis")
            plt.ylabel("Cumulative percentage of population")

            plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

sns.set(font_scale = 1, style="whitegrid")

for symptom in list_of_symptoms_incl_prescriptions:
    plt.figure(figsize=(10,8))

    df = df_dates_symptoms_ng12_cases_and_controls[df_dates_symptoms_ng12_cases_and_controls["symptom"]== symptom]
    df = df[df["dataset"]=="111"]

    if df.shape[0]>0:
        ax = sns.lineplot(x="days_between_activity_diagnosis",
                        y="cumulative_percentage",
                        data = df,
                        hue = "population" )
        
        plt.xticks(rotation=90)
        ax.invert_xaxis()

        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.title(symptom)
        plt.xlabel("Days before cancer diagnosis")
        plt.ylabel("Cumulative percentage of population")

        plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## where are symptoms being reported?

# COMMAND ----------

# patient - symptom - setting - flag

# cross join for base dataframe with days, dataset, ID, count

df_base_table = patient_pathways_utils.create_base_table_from_lists(list1=sources_of_flags,
                                                                    name1="source",
                                                                    list2=nice_guidelines_symptoms,
                                                                    name2="symptom",
                                                                    list3=unique_ids,
                                                                    name3="Patient_ID")
list_nice_guidelines_symptoms_flags = []

for symptom in nice_guidelines_symptoms:
    for col in df_pd_patient_flags.columns:
        if symptom in col and "flag" in col:
            list_nice_guidelines_symptoms_flags.append(col)

df_symptom_flags_long = pd.melt(df_pd_patient_flags, id_vars = "Patient_ID", value_vars = list_nice_guidelines_symptoms_flags, var_name = "col_name", value_name = "flag").fillna(0)

df_symptom_flags_long["symptom"] = df_symptom_flags_long["col_name"].apply(lambda x:x.split("flag_")[1]) 
df_symptom_flags_long["symptom"] = df_symptom_flags_long["symptom"].apply(lambda x:x.split("_reported_in")[0])

df_symptom_flags_long["source"] = df_symptom_flags_long["col_name"].apply(lambda x:x.split("_reported_in_")[1]) 
df_symptom_flags_long["source"] = df_symptom_flags_long["source"].apply(lambda x:x.split("_in_last")[0])

df_symptom_flags_long["source"] = np.where(~df_symptom_flags_long["source"].isin(sources_of_flags), "overall", df_symptom_flags_long["source"]) 
df_symptom_flags_long = df_base_table.merge(df_symptom_flags_long, on = ["Patient_ID", "symptom", "source"], how="left").fillna(0)
df_symptom_flags_long

# COMMAND ----------

sns.set(font_scale = 1, style="white")

for symptom in nice_guidelines_symptoms:

    # identify proportion/percentage of patients which reported this symptom at each source
    df = df_symptom_flags_long[df_symptom_flags_long["symptom"]==symptom].groupby("source")["flag"].value_counts(normalize=True).to_frame()
    df.columns = ["proportion"]
    df = df.reset_index()
    df = df[df["flag"]==0]
    df["percentage"] = 100-df["proportion"]*100

    if df.shape[0]>0:
        plt.figure(figsize=(10,8))
        sns.barplot(data = df, x = "source", y="percentage")
        plt.title(symptom)
        plt.xlabel("source")
        plt.ylabel("Percentage of overall population (%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## multiple symptoms across different sources

# COMMAND ----------

df_symptoms_count_by_source = df_symptom_flags_long.groupby(["Patient_ID", "source"])["flag"].sum().to_frame()
df_symptoms_count_by_source.columns=["count_symptoms_at_source"]
df_symptoms_count_by_source = df_symptoms_count_by_source.reset_index()
df_symptoms_count_by_source["flag_at_least_one_symptom_at_source"] = np.where(df_symptoms_count_by_source["count_symptoms_at_source"]>=1,1,0)
df_diff_sources = df_symptoms_count_by_source.groupby("Patient_ID")["flag_at_least_one_symptom_at_source"].sum()
df_diff_sources = df_diff_sources.to_frame()
df_diff_sources.columns= ["number_of_sources_with_symptoms_reported"]
df_diff_sources = df_diff_sources.reset_index()

df_count_number_sources_overall = patient_pathways_utils.count_and_pct(df_diff_sources,  col = "number_of_sources_with_symptoms_reported")
df_count_number_sources_overall

# COMMAND ----------

plt.figure(figsize=(10,8))
sns.set(font_scale = 1, style="white")
sns.barplot(data = df_count_number_sources_overall, x = "number_of_sources_with_symptoms_reported", y="percentage")

plt.xlabel("Number of sources (gp, acute, ecds, 111) where symptom is reported")
plt.ylabel("Percentage of overall population (%)")

# COMMAND ----------

df_count_symptoms_by_source_per_patient = df_symptom_flags_long.groupby(["Patient_ID", "symptom"])["flag"].sum().to_frame()
df_count_symptoms_by_source_per_patient.columns = ["count_of_sources"]
df_count_symptoms_by_source_per_patient = df_count_symptoms_by_source_per_patient.reset_index()

df_count_symptoms_by_source_per_patient_wide = df_count_symptoms_by_source_per_patient.pivot_table(index="Patient_ID", columns = "symptom", values = "count_of_sources")

df_count_symptoms_by_source_per_patient_wide

# COMMAND ----------

df_count_symptoms_per_patient = df_count_symptoms_by_source_per_patient.groupby("Patient_ID")["count_of_sources"].sum().to_frame()
df_count_symptoms_per_patient.columns=["total_num_symptoms"]
df_count_symptoms_per_patient = df_count_symptoms_per_patient.reset_index()
df_count_symptoms_per_patient

# COMMAND ----------

# MAGIC %md
# MAGIC ## PPV/Sensitivity

# COMMAND ----------

# calculate sensitivity and PPV per symptom, per source
df_metric_ratio_case_controls = df_dates_symptoms_ng12_cases_and_controls[df_dates_symptoms_ng12_cases_and_controls["days_between_activity_diagnosis"]==0].pivot_table(index=["col_name", "symptom", "dataset"], columns = "population", values = "cumulative_percentage").reset_index()

df_metric_ratio_case_controls["difference_in_pct_between_cases_and_controls"] = df_metric_ratio_case_controls["Cases"] - df_metric_ratio_case_controls["Controls"]
df_metric_ratio_case_controls["ratio"] = df_metric_ratio_case_controls["Cases"]/df_metric_ratio_case_controls["Controls"]

df_metric_ratio_case_controls

# COMMAND ----------

sns.set(font_scale = 1, style="whitegrid")

for symptom in list_of_symptoms_incl_prescriptions:
    plt.figure(figsize=(8,8))

    df = df_metric_ratio_case_controls[df_metric_ratio_case_controls["symptom"]==symptom]

    ax = sns.scatterplot(x="Cases",
                    y="difference_in_pct_between_cases_and_controls",
                    data = df,
                    hue = "dataset",
                    hue_order = ["gp", "111", "ecds", "acute", "overall"],
                    palette=pal_dataset,
                    s=75)

    plt.xlabel("Overall incidence in Lung Cancer population (%)")
    plt.ylabel("Difference in incidence between lung cancer cases and control (%)")
    plt.title(symptom)

    plt.xlim(xmin = 0)
    plt.ylim(ymin = -0.2)

    plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

df = df_metric_ratio_case_controls[(df_metric_ratio_case_controls["dataset"]=="overall") & (df_metric_ratio_case_controls["symptom"]=="NG12")]

plt.figure(figsize=(6,6))
sns.set(font_scale = 1, style="whitegrid")

ax = sns.scatterplot(x="Cases",
                y="ratio",
                data = df,
                hue = "col_name",
                style = "col_name",
                markers=True)

plt.xlabel("Overall incidence in Lung Cancer population (%)")
plt.ylabel("Ratio between cases and controls")
plt.legend(bbox_to_anchor=[1.05,1], loc=2)

plt.xlim(xmin = 0)
plt.ylim(ymin = 0)

# COMMAND ----------

df = df_metric_ratio_case_controls[(df_metric_ratio_case_controls["dataset"]=="overall") & (df_metric_ratio_case_controls["symptom"]!="NG12")]

plt.figure(figsize=(6,6))
sns.set(font_scale = 1, style="whitegrid")

dict_markers = {}

i=0
for symptom in df["symptom"].unique():
    dict_markers[symptom] = markers[i]
    i = i+1

ax = sns.scatterplot(x="Cases",
                y="ratio",
                data = df,
                hue = "symptom", 
                style="symptom",
                markers=True,
                palette=pal_symptom,
                s=75)

plt.xlabel("Overall incidence in Lung Cancer population (%)")
plt.ylabel("Ratio of incidence between cases and controls")
plt.legend(bbox_to_anchor=[1.05,1], loc=2)

plt.xlim(xmin = 0)
plt.ylim(ymin = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC # Metric 1 - Time to diagnosis/ chest x-ray between symptoms and cancer diagnosis (broken down by early/late diagnosis or deprivation)

# COMMAND ----------

# MAGIC %md
# MAGIC ## NG12 guidelines 

# COMMAND ----------

df_pd_patient_flags[cols_time_diagnosis_NG12].describe()

# COMMAND ----------

df_ng12_time_to_diagnosis = pd.melt(df_pd_patient_flags,
                                    id_vars = ["Patient_ID", "tumour_stage_group", "imd_decile_group", "Sex","route_earliest"],
                                    value_vars = cols_time_diagnosis_NG12).dropna(subset="value")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Crtitical symptom
# MAGIC The analysis in the following subsection is for patients who reported a critical symptom prior to their lung cancer diagnosis

# COMMAND ----------

# only those with critical symptom and x-ray

ids_critical_symptom_and_xray = df_pd_patient_flags[~df_pd_patient_flags["days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days"].isnull()]["Patient_ID"]
df = df_ng12_time_to_diagnosis[df_ng12_time_to_diagnosis["Patient_ID"].isin(ids_critical_symptom_and_xray)]

print(df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_critical_symptom_and_xray)][["days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days","time_diagnosis_for_one_critical_symptom"]].describe())

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days","time_diagnosis_for_one_critical_symptom",
                     ])
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.show()

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days","time_diagnosis_for_one_critical_symptom",
                     ],
            hue = "tumour_stage_group",
            hue_order = ["early", "late"],
            boxprops=dict(alpha=0.5),
            palette=pal_stage)
plt.title("Days Between NG12 condition and xray")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)
plt.show()

# COMMAND ----------

variables_for_quartile_analysis = ["tumour_stage_group", "imd_decile_group", "Sex","route_earliest", "Smoking_Flag", "LTC_COPD", var_mh] + list_of_mh_fields

df_patients_critical_symptom = df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_critical_symptom_and_xray)].copy()
df_patients_critical_symptom["quartile"] = pd.qcut(df_patients_critical_symptom['days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days'], q=[0, .25, .5, .75, 1.], labels = ["<25", "25-50", "50-75", ">75"])

display(df_patients_critical_symptom.groupby("quartile")["days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days"].describe())

for metric in variables_for_quartile_analysis:

    df_summary = patient_pathways_utils.count_and_pct(df_patients_critical_symptom, metric, "quartile")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric + ": quartiles by time to chest imaging from reporting of critical symptom")
    sns.set(font_scale = 1, style="white")
    sns.barplot(x="quartile", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

df_patients_critical_symptom_quartiles_activity = pd.DataFrame()

for quartile in df_patients_critical_symptom["quartile"].unique():
    df = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_patients_critical_symptom[df_patients_critical_symptom["quartile"]==quartile],
                                                                                           date_columns_to_process = date_columns_to_process,
                                                                                           id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                                           diagnosis_date_col = "diagnosis_date_earliest",
                                                                                           history_days = history_days)

    df["quartile"] = quartile

    df_patients_critical_symptom_quartiles_activity = pd.concat([df_patients_critical_symptom_quartiles_activity,df], axis =0 )

df_patients_critical_symptom_quartiles_activity["symptom"] = df_patients_critical_symptom_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_patients_critical_symptom_quartiles_activity["dataset"] = df_patients_critical_symptom_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

sns.set(font_scale = 1.5, style="whitegrid")

for col_name in df_patients_critical_symptom_quartiles_activity["col_name"].unique():
    plt.figure(figsize=(10,8))   

    df = df_patients_critical_symptom_quartiles_activity[df_patients_critical_symptom_quartiles_activity["col_name"]== col_name]
    df = df[df["dataset"]=="overall"]

    if df.shape[0]>0:

        ax = sns.lineplot(x="days_between_activity_diagnosis",
                        y="cumulative_percentage",
                        data = df,
                        hue = "quartile",
                        hue_order = ["<25", "25-50", "50-75", ">75"],
                        style="quartile")
        
        plt.xticks(rotation=90)
        ax.invert_xaxis()

        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.title(col_name)
        plt.xlabel("Days before cancer diagnosis")
        plt.ylabel("Cumulative percentage of population")

        plt.legend(bbox_to_anchor=[1.05,1], loc=2)


# COMMAND ----------

df_patient_with_critical_symptom = df_pd_patient_flags[df_pd_patient_flags["flag_atleast_one_critical_symptoms_NICE_guidelines_in_last_365_days"]==1].copy()
print(df_patient_with_critical_symptom["flag_chest_xray_in_last_365_days"].value_counts())

for metric in variables_for_quartile_analysis:

    df_summary = patient_pathways_utils.count_and_pct(df_patient_with_critical_symptom, metric, "flag_chest_xray_in_last_365_days")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric)
    sns.set(font_scale = 1, style="white")
    sns.barplot(x="flag_chest_xray_in_last_365_days", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### investigate quartile based on time to diagnosis

# COMMAND ----------

df_patients_critical_symptom_to_diagnosis = df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_critical_symptom_and_xray)].copy()
df_patients_critical_symptom_to_diagnosis["quartile"] = pd.qcut(df_patients_critical_symptom_to_diagnosis['time_diagnosis_for_one_critical_symptom'], q=[0, .25, .5, .75, 1.], labels=["<25", "25-50", "50-75", ">75"])

display(df_patients_critical_symptom_to_diagnosis.groupby("quartile")["time_diagnosis_for_one_critical_symptom"].describe())

for metric in variables_for_quartile_analysis:

    df_summary = patient_pathways_utils.count_and_pct(df_patients_critical_symptom_to_diagnosis, metric, "quartile")
    display(df_summary)



    plt.figure(figsize=(12,6))
    plt.title(metric + ": quartiles by time to diagnosis from reporting of critical symptom")
    sns.set(font_scale = 1, style="white")
    sns.barplot(x= "quartile", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Smoker with 1 symptom
# MAGIC The analysis in the following subsection is for lung cancer patients who were smokers and reported an unexplained symptom in the 1 year before their cancer diagnosis

# COMMAND ----------

# only smokers with 1 symptom and x-ray

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

ids_one_symptom_smoker_and_xray = df_pd_patient_flags[~df_pd_patient_flags["days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days"].isnull()]["Patient_ID"]
df = df_ng12_time_to_diagnosis[df_ng12_time_to_diagnosis["Patient_ID"].isin(ids_one_symptom_smoker_and_xray)]

print(df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_one_symptom_smoker_and_xray)][["days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days", "time_diagnosis_for_smoker_with_unexplained_symptoms"]].describe())

sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days", "time_diagnosis_for_smoker_with_unexplained_symptoms",
                     ])
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.show()

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days", "time_diagnosis_for_smoker_with_unexplained_symptoms",
                     ],
            hue = "tumour_stage_group",
            hue_order = ["early", "late"],
            boxprops=dict(alpha=0.5),
            palette=pal_stage)
plt.title("Days Between NG12 condition and xray")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)
plt.show()

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days", "time_diagnosis_for_smoker_with_unexplained_symptoms",
                     ],
            hue = "imd_decile_group",
            boxprops=dict(alpha=0.5))
plt.title("Days Between NG12 condition and xray")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)
plt.show()

# COMMAND ----------

df_patients_smokers_one_symptom = df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_one_symptom_smoker_and_xray)].copy()
df_patients_smokers_one_symptom["quartile"] = pd.qcut(df_patients_smokers_one_symptom['days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days'], q=[0, .25, .5, .75, 1.], labels=["<25", "25-50", "50-75", ">75"])

display(df_patients_smokers_one_symptom.groupby("quartile")["days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days"].describe())

for metric in variables_for_quartile_analysis + ["flag_inhaler_reported_in_last_365_days", "flag_oral_antibiotic_reported_in_last_365_days"]:

    df_summary = patient_pathways_utils.count_and_pct(df_patients_smokers_one_symptom, metric, "quartile")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric + ": quartiles by time to chest imaging from reporting of unexplained symptom for smoker")
    sns.set(font_scale = 1, style="white")
    sns.barplot(x= "quartile", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC Plot activity leading up to chest imaging

# COMMAND ----------

df_patients_smokers_one_symptom["date_of_xray_after_symptom"] = df_patients_smokers_one_symptom["earliest_date_one_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days"] + pd.to_timedelta(df_patients_smokers_one_symptom['days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days'], unit='d')

df_smokers_quartiles_activity = pd.DataFrame()

for quartile in df_patients_smokers_one_symptom["quartile"].unique():
    df = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_patients_smokers_one_symptom[df_patients_smokers_one_symptom["quartile"]==quartile],
                                                                                           date_columns_to_process = date_columns_to_process,
                                                                                           id_vars = ["Patient_ID", "tumour_stage_group", "date_of_xray_after_symptom"],
                                                                                           diagnosis_date_col = "date_of_xray_after_symptom",
                                                                                           history_days = history_days)

    df["quartile"] = quartile

    df_smokers_quartiles_activity = pd.concat([df_smokers_quartiles_activity,df], axis =0 )

df_smokers_quartiles_activity["symptom"] = df_smokers_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_smokers_quartiles_activity["dataset"] = df_smokers_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

sns.set(font_scale = 1.5, style="whitegrid")

for col_name in df_smokers_quartiles_activity["col_name"].unique():
    plt.figure(figsize=(10,8))   

    df = df_smokers_quartiles_activity[df_smokers_quartiles_activity["col_name"]== col_name]
    df = df[df["dataset"]=="overall"]

    if df.shape[0]>0:

        ax = sns.lineplot(x="days_between_activity_diagnosis",
                        y="cumulative_percentage",
                        data = df,
                        hue = "quartile",
                        hue_order = ["<25", "25-50", "50-75", ">75"],
                        style="quartile")
        
        plt.xticks(rotation=90)
        ax.invert_xaxis()

        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.title(col_name)
        plt.xlabel("Days before chest imaging")
        plt.ylabel("Cumulative percentage of population")

        plt.legend(bbox_to_anchor=[1.05,1], loc=2)


# COMMAND ----------

# MAGIC %md
# MAGIC Plot activity leading up to diagnosis

# COMMAND ----------

df_smokers_quartiles_activity = pd.DataFrame()

for quartile in df_patients_smokers_one_symptom["quartile"].unique():
    df = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_patients_smokers_one_symptom[df_patients_smokers_one_symptom["quartile"]==quartile],
                                                                                           date_columns_to_process = date_columns_to_process,
                                                                                           id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                                           diagnosis_date_col = "diagnosis_date_earliest",
                                                                                           history_days = history_days)

    df["quartile"] = quartile

    df_smokers_quartiles_activity = pd.concat([df_smokers_quartiles_activity,df], axis =0 )


# those with no xray
df_no_xray = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[(df_pd_patient_flags["flag_smoker_with_unexplained_symptoms_in_last_365_days"]==1) & (df_pd_patient_flags["flag_chest_xray_in_last_365_days"]==0) ],
                                                                                        date_columns_to_process = date_columns_to_process,
                                                                                        id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                                        diagnosis_date_col = "diagnosis_date_earliest",
                                                                                        history_days = history_days)

df_no_xray["quartile"] = "no xray"

df_smokers_quartiles_activity = pd.concat([df_smokers_quartiles_activity,df_no_xray], axis =0 )

df_smokers_quartiles_activity["symptom"] = df_smokers_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_smokers_quartiles_activity["dataset"] = df_smokers_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

sns.set(font_scale = 1.5, style="whitegrid")

for col_name in df_smokers_quartiles_activity["col_name"].unique():
    plt.figure(figsize=(10,8))   

    df = df_smokers_quartiles_activity[df_smokers_quartiles_activity["col_name"]== col_name]
    df = df[df["dataset"]=="overall"]

    if df.shape[0]>0:

        ax = sns.lineplot(x="days_between_activity_diagnosis",
                        y="cumulative_percentage",
                        data = df,
                        hue = "quartile",
                        hue_order = ["<25", "25-50", "50-75", ">75", "no xray"],
                        style="quartile")
        
        plt.xticks(rotation=90)
        ax.invert_xaxis()

        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.title(col_name)
        plt.xlabel("Days before cancer diagnosis")
        plt.ylabel("Cumulative percentage of population")

        plt.legend(bbox_to_anchor=[1.05,1], loc=2)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Investigate quartiles based on time to diagnosis

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.set(font_scale = 1.25, style="white")
sns.scatterplot(x= "time_diagnosis_for_smoker_with_unexplained_symptoms", y="days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days", data = df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_one_symptom_smoker_and_xray)], hue = "tumour_stage_group", palette=pal_stage, hue_order = ["early", "late", "unknown"], alpha = 0.75)
plt.xlabel("Days between symptom and diagnosis")
plt.ylabel("Days between symptom and xray")

plt.figure(figsize=(10,10))
sns.set(font_scale = 1.25, style="white")
sns.scatterplot(x= "time_diagnosis_for_smoker_with_unexplained_symptoms", y="days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days", data = df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_one_symptom_smoker_and_xray)], hue = "route_earliest", palette=pal_route, alpha = 0.75)
plt.xlabel("Days between symptom and diagnosis")
plt.ylabel("Days between symptom and xray")


# COMMAND ----------

df_patients_smokers_one_symptom_to_diagnosis = df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_one_symptom_smoker_and_xray)].copy()
df_patients_smokers_one_symptom_to_diagnosis["quartile"] = pd.qcut(df_patients_smokers_one_symptom_to_diagnosis['time_diagnosis_for_smoker_with_unexplained_symptoms'], q=[0, .25, .5, .75, 1.], labels=["<25", "25-50", "50-75", ">75"])

display(df_patients_smokers_one_symptom_to_diagnosis.groupby("quartile")["time_diagnosis_for_smoker_with_unexplained_symptoms"].describe())

for metric in variables_for_quartile_analysis:

    df_summary = patient_pathways_utils.count_and_pct(df_patients_smokers_one_symptom_to_diagnosis, metric, "quartile")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric + ": quartiles by time to diagnosis from reporting of unexplained symptom for smoker")
    sns.set(font_scale = 1, style="white")
    sns.barplot(x= "quartile", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

df_smokers_quartiles_activity = pd.DataFrame()

for quartile in df_patients_smokers_one_symptom["quartile"].unique():
    df = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_patients_smokers_one_symptom_to_diagnosis[df_patients_smokers_one_symptom_to_diagnosis["quartile"]==quartile],
                                                                                           date_columns_to_process = date_columns_to_process,
                                                                                           id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                                           diagnosis_date_col = "diagnosis_date_earliest",
                                                                                           history_days = history_days)

    df["quartile"] = quartile

    df_smokers_quartiles_activity = pd.concat([df_smokers_quartiles_activity,df], axis =0 )

# those with no xray
df_no_xray = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[(df_pd_patient_flags["flag_smoker_with_unexplained_symptoms_in_last_365_days"]==1) & (df_pd_patient_flags["flag_chest_xray_in_last_365_days"]==0) ],
                                                                                        date_columns_to_process = date_columns_to_process,
                                                                                        id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                                        diagnosis_date_col = "diagnosis_date_earliest",
                                                                                        history_days = history_days)

df_no_xray["quartile"] = "no xray"

df_smokers_quartiles_activity = pd.concat([df_smokers_quartiles_activity,df_no_xray], axis =0 )

df_smokers_quartiles_activity["symptom"] = df_smokers_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_smokers_quartiles_activity["dataset"] = df_smokers_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

sns.set(font_scale = 1.5, style="whitegrid")

for col_name in df_smokers_quartiles_activity["col_name"].unique():
    plt.figure(figsize=(10,8))   

    df = df_smokers_quartiles_activity[df_smokers_quartiles_activity["col_name"]== col_name]
    df = df[df["dataset"]=="overall"]

    if df.shape[0]>0:

        ax = sns.lineplot(x="days_between_activity_diagnosis",
                        y="cumulative_percentage",
                        data = df,
                        hue = "quartile",
                        hue_order = ["<25", "25-50", "50-75", ">75", "no xray"],
                        style="quartile")
        
        plt.xticks(rotation=90)
        ax.invert_xaxis()

        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.title(col_name)
        plt.xlabel("Days before cancer diagnosis")
        plt.ylabel("Cumulative percentage of population")

        plt.legend(bbox_to_anchor=[1.05,1], loc=2)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Compare those who got an xray from those who didnt

# COMMAND ----------

df_smoker_with_symptom = df_pd_patient_flags[df_pd_patient_flags["flag_smoker_with_unexplained_symptoms_in_last_365_days"]==1].copy()
df_smoker_with_symptom["flag_chest_xray_in_last_365_days"].value_counts()

for metric in variables_for_quartile_analysis:

    df_summary = patient_pathways_utils.count_and_pct(df_smoker_with_symptom, metric, "flag_chest_xray_in_last_365_days")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric)
    sns.set(font_scale = 1, style="white")
    sns.barplot(x="flag_chest_xray_in_last_365_days", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### compare quarters for time to x-ray/diagnosis

# COMMAND ----------

df_patients_smokers_one_symptom_compare_quartiles = df_patients_smokers_one_symptom.merge(df_patients_smokers_one_symptom_to_diagnosis[["Patient_ID", "quartile"]], on = "Patient_ID", suffixes = ("_time_imaging", "_time_diagnosis"))

df_patients_smokers_one_symptom_compare_quartiles.groupby(["quartile_time_imaging"])["quartile_time_diagnosis"].value_counts()

# COMMAND ----------

# fast to x-ray, fast to diagnosis
df = df_patients_smokers_one_symptom_compare_quartiles[(df_patients_smokers_one_symptom_compare_quartiles["quartile_time_imaging"].isin(["<25", "25-50"])) & 
                                                  (df_patients_smokers_one_symptom_compare_quartiles["quartile_time_diagnosis"].isin(["<25", "25-50"]))]

print(df.shape[0])
for metric in variables_for_quartile_analysis:
    
    df_summary = patient_pathways_utils.count_and_pct(df, metric)
    display(df_summary)



# COMMAND ----------

# fast to x-ray, slow to diagnosis
df = df_patients_smokers_one_symptom_compare_quartiles[(df_patients_smokers_one_symptom_compare_quartiles["quartile_time_imaging"].isin(["<25", "25-50"])) & 
                                                  (df_patients_smokers_one_symptom_compare_quartiles["quartile_time_diagnosis"].isin(["50-75", ">75"]))]

print(df.shape[0])

for metric in variables_for_quartile_analysis:
    
    df_summary = patient_pathways_utils.count_and_pct(df, metric)
    display(df_summary)


# COMMAND ----------

# slow to x-ray, faster to diagnosis
df = df_patients_smokers_one_symptom_compare_quartiles[(df_patients_smokers_one_symptom_compare_quartiles["quartile_time_imaging"].isin(["50-75", ">75"])) & 
                                                  (df_patients_smokers_one_symptom_compare_quartiles["quartile_time_diagnosis"].isin(["<25", "25-50"]))]

print(df.shape[0])
for metric in variables_for_quartile_analysis:
    
    df_summary = patient_pathways_utils.count_and_pct(df, metric)
    display(df_summary)


# COMMAND ----------

# slow to x-ray, slow to diagnosis
df = df_patients_smokers_one_symptom_compare_quartiles[(df_patients_smokers_one_symptom_compare_quartiles["quartile_time_imaging"].isin(["50-75", ">75"])) & 
                                                  (df_patients_smokers_one_symptom_compare_quartiles["quartile_time_diagnosis"].isin(["50-75", ">75"]))]

print(df.shape[0])
for metric in variables_for_quartile_analysis:
    
    df_summary = patient_pathways_utils.count_and_pct(df, metric)
    display(df_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Two NG12 symptoms
# MAGIC
# MAGIC The analysis in the following subsection is for lung cancer patients who reported two or more unexplained symptoms in the 1 year before their lung cancer diagnosis

# COMMAND ----------

# only those with 2 symptoms

ids_two_symptoms_and_xray = df_pd_patient_flags[~df_pd_patient_flags["days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days"].isnull()]["Patient_ID"]
df = df_ng12_time_to_diagnosis[df_ng12_time_to_diagnosis["Patient_ID"].isin(ids_two_symptoms_and_xray)]

print(df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_two_symptoms_and_xray)][["days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days", "time_diagnosis_for_two_or_more_unexplained_symptoms"]].describe())

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days", "time_diagnosis_for_two_or_more_unexplained_symptoms",
                     ])
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days", "time_diagnosis_for_two_or_more_unexplained_symptoms",
                     ],
            hue = "tumour_stage_group",
            hue_order = ["early", "late"],
            boxprops=dict(alpha=0.5),
            palette=pal_stage)
plt.title("Days Between NG12 condition and xray")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)
plt.show()

# COMMAND ----------

df_patients_two_symptoms = df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_two_symptoms_and_xray)].copy()
df_patients_two_symptoms["quartile"] = pd.qcut(df_patients_two_symptoms['days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days'], q=[0, .25, .5, .75, 1.], labels=["<25", "25-50", "50-75", ">75"])
print(df_patients_two_symptoms.shape)

display(df_patients_two_symptoms.groupby("quartile")["days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days"].describe())


for metric in variables_for_quartile_analysis + ["flag_inhaler_reported_in_last_365_days", "flag_oral_antibiotic_reported_in_last_365_days"]:

    df_summary = patient_pathways_utils.count_and_pct(df_patients_two_symptoms, metric, "quartile")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric + ": quartiles by time to chest imaging from reporting of two unexplained symptoms")
    sns.set(font_scale = 1, style="white")
    sns.barplot(x="quartile", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

df_two_symptoms_quartiles_activity = pd.DataFrame()

for quartile in df_patients_two_symptoms["quartile"].unique():
    df = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_patients_two_symptoms[df_patients_two_symptoms["quartile"]==quartile],
                                                                                           date_columns_to_process = date_columns_to_process,
                                                                                           id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                                           diagnosis_date_col = "diagnosis_date_earliest",
                                                                                           history_days = history_days)

    df["quartile"] = quartile

    df_two_symptoms_quartiles_activity = pd.concat([df_two_symptoms_quartiles_activity,df], axis =0 )

df_two_symptoms_quartiles_activity["symptom"] = df_two_symptoms_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_two_symptoms_quartiles_activity["dataset"] = df_two_symptoms_quartiles_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

sns.set(font_scale = 1.5, style="whitegrid")

for col_name in df_two_symptoms_quartiles_activity["col_name"].unique():
    plt.figure(figsize=(10,8))   

    df = df_two_symptoms_quartiles_activity[df_two_symptoms_quartiles_activity["col_name"]== col_name]
    df = df[df["dataset"]=="overall"]

    if df.shape[0]>0:

        ax = sns.lineplot(x="days_between_activity_diagnosis",
                        y="cumulative_percentage",
                        data = df,
                        hue = "quartile",
                        hue_order = ["<25", "25-50", "50-75", ">75"],
                        style="quartile")
        
        plt.xticks(rotation=90)
        ax.invert_xaxis()

        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.title(col_name)
        plt.xlabel("Days before cancer diagnosis")
        plt.ylabel("Cumulative percentage of population")

        plt.legend(bbox_to_anchor=[1.05,1], loc=2)


# COMMAND ----------

df_patient_with_two_symptoms = df_pd_patient_flags[df_pd_patient_flags["flag_two_or_more_unexplained_symptoms_in_last_365_days"]==1].copy()
display(df_patient_with_two_symptoms["flag_chest_xray_in_last_365_days"].value_counts())

for metric in variables_for_quartile_analysis:

    df_summary = patient_pathways_utils.count_and_pct(df_patient_with_two_symptoms, metric, "flag_chest_xray_in_last_365_days")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric)
    sns.set(font_scale = 1, style="white")
    sns.barplot(x="flag_chest_xray_in_last_365_days", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### investigate quartile based on time to diagnosis

# COMMAND ----------

df_patients_two_symptoms_to_diagnosis = df_pd_patient_flags[df_pd_patient_flags["Patient_ID"].isin(ids_two_symptoms_and_xray)].copy()
df_patients_two_symptoms_to_diagnosis["quartile"] = pd.qcut(df_patients_two_symptoms_to_diagnosis['time_diagnosis_for_two_or_more_unexplained_symptoms'], q=[0, .25, .5, .75, 1.], labels=["<25", "25-50", "50-75", ">75"])

display(df_patients_two_symptoms_to_diagnosis.groupby("quartile")["time_diagnosis_for_two_or_more_unexplained_symptoms"].describe())

for metric in variables_for_quartile_analysis:

    df_summary = patient_pathways_utils.count_and_pct(df_patients_two_symptoms_to_diagnosis, metric, "quartile")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric + ": quartiles by time to diagnosis from reporting of two unexplained symptoms")
    sns.set(font_scale = 1, style="white")
    sns.barplot(x= "quartile", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### plot breakdown by stage

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_ng12_time_to_diagnosis[~df_ng12_time_to_diagnosis["variable"].isin(["time_diagnosis_from_earliest_xray", "time_diagnosis_from_latest_xray"])],
            y = "variable",
            x="value",
            order = ["days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days","time_diagnosis_for_one_critical_symptom",
                     "days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days", "time_diagnosis_for_smoker_with_unexplained_symptoms",
                     "days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days", "time_diagnosis_for_two_or_more_unexplained_symptoms",
                     ])
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.show()

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_ng12_time_to_diagnosis[~df_ng12_time_to_diagnosis["variable"].isin(["time_diagnosis_from_earliest_xray", "time_diagnosis_from_latest_xray"])],
            y = "variable",
            x="value",
            order = [
                     "time_diagnosis_for_one_critical_symptom",
                     "time_diagnosis_for_smoker_with_unexplained_symptoms",
                     "time_diagnosis_for_two_or_more_unexplained_symptoms",],
            hue="tumour_stage_group",
            hue_order = ["early", "late"],
            boxprops=dict(alpha=0.5),
            palette=pal_stage)
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)
plt.show()

# COMMAND ----------

df = df_ng12_time_to_diagnosis[df_ng12_time_to_diagnosis["variable"].isin([
                     "time_diagnosis_for_one_critical_symptom",
                     "time_diagnosis_for_smoker_with_unexplained_symptoms",
                     "time_diagnosis_for_two_or_more_unexplained_symptoms"])]
df.groupby(["variable", "tumour_stage_group"])["value"].describe()

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

df = df_ng12_time_to_diagnosis[~df_ng12_time_to_diagnosis["variable"].isin(["time_diagnosis_from_earliest_xray", "time_diagnosis_from_latest_xray"])]
df = df[df["imd_decile_group"]!="unknown"]

sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days","time_diagnosis_for_one_critical_symptom",
                     "days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days", "time_diagnosis_for_smoker_with_unexplained_symptoms",
                     "days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days", "time_diagnosis_for_two_or_more_unexplained_symptoms",],
            hue="imd_decile_group",
            boxprops=dict(alpha=0.5),
            palette=pal_imd)
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)
plt.show()

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

df = df_ng12_time_to_diagnosis[~df_ng12_time_to_diagnosis["variable"].isin(["time_diagnosis_from_earliest_xray", "time_diagnosis_from_latest_xray"])]
df = df[df["imd_decile_group"]!="unknown"]

sns.boxplot(data=df,
            y = "variable",
            x="value",
            order = ["time_diagnosis_for_smoker_with_unexplained_symptoms", "days_xray_to_earliest_date_smoker_with_unexplained_symptoms_in_last_365_days",
                     "time_diagnosis_for_two_or_more_unexplained_symptoms", "days_xray_to_earliest_date_two_reported_unexplained_symptoms_NICE_guidelines_in_last_365_days",
                     "time_diagnosis_for_one_critical_symptom", "days_xray_to_date_first_reported_critical_symptoms_NICE_guidelines_in_last_365_days"],
            hue="imd_decile_group",
            boxprops=dict(alpha=0.5),
            palette=pal_imd)
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)
plt.show()

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_ng12_time_to_diagnosis, y = "variable", x="value")
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,10))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_ng12_time_to_diagnosis, y = "variable", x="value", hue="tumour_stage_group", hue_order = ["early", "late", "unknown"], palette = pal_stage,)
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

plt.figure(figsize=(16,20))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_ng12_time_to_diagnosis, y = "variable", x="value", hue="route_earliest", hue_order = ["GP referral", "USC", "Emergency presentation", "Other outpatient"])
plt.title("Days Between NG12 condition and Diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

list_time_to_diagnosis_from_symptom_columns = []

for symptom in list_of_symptoms_incl_prescriptions:
    for col in df_patient_flags.columns:
        if symptom in col and "time_diagnosis_" in col:
            list_time_to_diagnosis_from_symptom_columns.append(col)

df_symptoms_time_to_diagnosis = pd.melt(df_pd_patient_flags,
                                    id_vars = ["Patient_ID", "tumour_stage_group", "route_earliest", "Smoking_Flag"],
                                    value_vars = list_time_to_diagnosis_from_symptom_columns).dropna(subset="value")

df_symptoms_time_to_diagnosis["Smoking_Flag"] = df_symptoms_time_to_diagnosis["Smoking_Flag"].fillna(0)
df_symptoms_time_to_diagnosis["symptom"] = df_symptoms_time_to_diagnosis["variable"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG1"))
df_symptoms_time_to_diagnosis["dataset"] = df_symptoms_time_to_diagnosis["variable"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))


df_symptoms_time_to_diagnosis

# COMMAND ----------

for symptom in list_of_symptoms_incl_prescriptions:
    df = df_symptoms_time_to_diagnosis[df_symptoms_time_to_diagnosis["symptom"]==symptom]
    df = df[df["dataset"]!="overall"]

    if df.shape[0]>0:
        plt.figure(figsize=(12,6))
        sns.set(font_scale = 1, style="white")

        sns.boxplot(data=df, y = "dataset", x="value")
        plt.title(f"Days Between {symptom} and diagnosis")
        plt.ylabel("")
        plt.xlabel("Number of days")

# COMMAND ----------

plt.figure(figsize=(16,30))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_symptoms_time_to_diagnosis, y = "variable", x="value")
plt.title("Days Between symptom and diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

df = df_symptoms_time_to_diagnosis[df_symptoms_time_to_diagnosis["dataset"]=="overall"]

plt.figure(figsize=(16,30))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df, y = "variable", x="value", hue = "Smoking_Flag")
plt.title("Days Between symptom and diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### symptoms by venue : patients with only 2 symptoms overall

# COMMAND ----------

df_symptoms_events = patient_pathways_utils.create_long_event_table_from_columns(df_pd_patient_flags,
                                                            date_columns_to_process = date_columns_to_process,
                                                            id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                            diagnosis_date_col = "diagnosis_date_earliest",
                                                            history_days = history_days)

df_symptoms_events["symptom"] = df_symptoms_events["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_symptoms_events["dataset"] = df_symptoms_events["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))
df_symptoms_events = df_symptoms_events[df_symptoms_events["dataset"]!="overall"]

df_patients_with_only_two_symptoms = df_pd_patient_flags.merge(df_count_symptoms_per_patient[df_count_symptoms_per_patient["total_num_symptoms"]==2], on="Patient_ID", how="inner") 

df_symptoms_events = df_symptoms_events.merge(df_patients_with_only_two_symptoms[["Patient_ID"]], on = "Patient_ID")
df_symptoms_events.sort_values(by=["Patient_ID", "days_between_activity_diagnosis"], ascending = False)
df_symptoms_events["order"] = df_symptoms_events.groupby("Patient_ID")["symptom"].cumcount()+1
df_symptoms_events["symptom_source"] = df_symptoms_events["symptom"] + "_" + df_symptoms_events["dataset"]
df_symptoms_events


# COMMAND ----------

df_symptoms_events_wide = df_symptoms_events.pivot(index="Patient_ID", columns = "order", values = "symptom_source").reset_index()
df_combos = df_symptoms_events_wide.groupby([1,2])["Patient_ID"].count().to_frame()

df_combos.columns=["count"]
df_combos=df_combos.reset_index()
df_combos.sort_values(by="count", ascending = False).iloc[0:20]

# COMMAND ----------

# MAGIC %md
# MAGIC ### symptoms by venue : patients with only 1 symptoms overall

# COMMAND ----------

df_symptoms_time_to_diagnosis_one_symptom_only = df_symptoms_time_to_diagnosis.merge(df_count_symptoms_per_patient[df_count_symptoms_per_patient["total_num_symptoms"]==1], 
                                                                                    on="Patient_ID", 
                                                                                    how="inner")
df_symptoms_time_to_diagnosis_one_symptom_only = df_symptoms_time_to_diagnosis_one_symptom_only[df_symptoms_time_to_diagnosis_one_symptom_only["symptom"].isin(nice_guidelines_symptoms)] 
df_symptoms_time_to_diagnosis_one_symptom_only

# COMMAND ----------

df_count_symptom_per_dataset = df_symptoms_time_to_diagnosis_one_symptom_only.groupby(["symptom", "dataset"])["Patient_ID"].count()
df_count_symptom_per_dataset = df_count_symptom_per_dataset.to_frame()
df_count_symptom_per_dataset = df_count_symptom_per_dataset.reset_index()
df_count_symptom_per_dataset

# COMMAND ----------

plt.figure(figsize=(16,30))
sns.set(font_scale = 1, style="white")
df = df_symptoms_time_to_diagnosis_one_symptom_only[df_symptoms_time_to_diagnosis_one_symptom_only["symptom"].isin(nice_guidelines_symptoms)] 
df = df[df["dataset"]!="overall"]
sns.boxplot(data=df, y = "symptom", x="value", hue = "dataset", hue_order = ["gp", "111", "acute", "ecds"], palette = pal_dataset)
plt.title("Days Between symptom and diagnosis")
plt.ylabel("")
plt.xlabel("Number of days")
plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## percentage who met criteria

# COMMAND ----------

df_combined_patient_flags = pd.concat([df_pd_patient_flags, df_pd_patient_flags_control], axis = 0)

# COMMAND ----------

cols_for_flags = [f"flag_smoker_with_unexplained_symptoms_in_last_{str(history_days)}_days",
                  f"flag_atleast_one_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days",
                  f"flag_two_or_more_unexplained_symptoms_in_last_{str(history_days)}_days",]


# COMMAND ----------

for col in cols_for_flags:
    print(col)
    print("Percentage in lung cancer population: ", 100*df_pd_patient_flags[df_pd_patient_flags[col]==1].shape[0]/df_pd_patient_flags.shape[0])
    print("Percentage in control population: ", 100*df_pd_patient_flags_control[df_pd_patient_flags_control[col]==1].shape[0]/df_pd_patient_flags_control.shape[0])

# COMMAND ----------

col = f"number_of_unexplained_symptoms_NICE_guidelines_in_last_{str(history_days)}_days"
print(col)
print("Percentage in lung cancer population: ", 100*df_pd_patient_flags[df_pd_patient_flags[col]>=1].shape[0]/df_pd_patient_flags.shape[0])
print("Percentage in control population: ", 100*df_pd_patient_flags_control[df_pd_patient_flags_control[col]>=1].shape[0]/df_pd_patient_flags_control.shape[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## percentage who had chest x-ray
# MAGIC

# COMMAND ----------

# met criteria for chest x-ray
# had chest x-ray

df_pd_patient_flags["flag_met_condition_for_xray"] = np.where(
    (df_pd_patient_flags[f"flag_two_or_more_unexplained_symptoms_in_last_{str(history_days)}_days"]==1) | 
    (df_pd_patient_flags[f"flag_smoker_with_unexplained_symptoms_in_last_{str(history_days)}_days"] ==1) | 
    (df_pd_patient_flags[f"flag_atleast_one_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days"] == 1),
    1,
    0)

# COMMAND ----------

# lung cancer population
df_pd_patient_flags[f"flag_chest_xray_in_last_{history_days}_days"].value_counts()

# COMMAND ----------

# lung cancer population
df_pd_patient_flags[f"flag_chest_xray_in_last_{history_days}_days"].value_counts(normalize=True)

# COMMAND ----------

# control population
df_pd_patient_flags_control[f"flag_chest_xray_in_last_{history_days}_days"].value_counts(normalize=True)

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, f"flag_chest_xray_in_last_{history_days}_days", "flag_met_condition_for_xray")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, f"flag_chest_xray_in_last_{history_days}_days", f"flag_two_or_more_unexplained_symptoms_in_last_{str(history_days)}_days")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, f"flag_chest_xray_in_last_{history_days}_days", f"flag_smoker_with_unexplained_symptoms_in_last_{str(history_days)}_days")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags,  f"flag_atleast_one_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days", f"flag_chest_xray_in_last_{history_days}_days")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, f"flag_chest_xray_in_last_{history_days}_days", f"flag_atleast_one_critical_symptoms_NICE_guidelines_in_last_{str(history_days)}_days")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, f"flag_chest_xray_in_last_{history_days}_days", "route_earliest")

# COMMAND ----------

# compare stage of those who had chest x-ray and those who didn't
patient_pathways_utils.count_and_pct(df_pd_patient_flags, "tumour_stage_group",f"flag_chest_xray_in_last_{history_days}_days" )


# COMMAND ----------

display(df_pd_patient_flags["flag_chest_xray_in_last_365_days"].value_counts())

for metric in ["tumour_stage_group", "imd_decile_group", "Sex","route_earliest", "Smoking_Flag", "LTC_COPD"]:

    df_summary = patient_pathways_utils.count_and_pct(df_pd_patient_flags, metric, "flag_chest_xray_in_last_365_days")
    display(df_summary)

    plt.figure(figsize=(12,6))
    plt.title(metric)
    sns.set(font_scale = 1, style="white")
    sns.barplot(x="flag_chest_xray_in_last_365_days", y="percentage", hue=metric, data = df_summary)
    plt.legend(bbox_to_anchor=[1.05,1], loc = 2)

# COMMAND ----------

df_with_without_xray_activity = pd.DataFrame()

for xray_status in df_pd_patient_flags["flag_chest_xray_in_last_365_days"].unique():
    df = patient_pathways_utils.create_cumulative_count_percentage_from_date_col(df_pd_patient_flags[df_pd_patient_flags["flag_chest_xray_in_last_365_days"]==xray_status],
                                                                                           date_columns_to_process = date_columns_to_process,
                                                                                           id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                                                                           diagnosis_date_col = "diagnosis_date_earliest",
                                                                                           history_days = history_days)

    df["population"] = xray_status

    df_with_without_xray_activity = pd.concat([df_with_without_xray_activity,df], axis =0 )

df_with_without_xray_activity["symptom"] = df_with_without_xray_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,list_of_symptoms_incl_prescriptions, null_return = "NG12"))
df_with_without_xray_activity["dataset"] = df_with_without_xray_activity["col_name"].apply(lambda col_name : identify_val_in_list(col_name,["gp", "acute", "111", "ecds"], null_return = "overall"))

sns.set(font_scale = 1.5, style="whitegrid")

for col_name in df_with_without_xray_activity["col_name"].unique():
    plt.figure(figsize=(10,8))   

    df = df_with_without_xray_activity[df_with_without_xray_activity["col_name"]== col_name]
    df = df[df["dataset"]=="overall"]

    if df.shape[0]>0:

        ax = sns.lineplot(x="days_between_activity_diagnosis",
                        y="cumulative_percentage",
                        data = df,
                        hue = "population",
                        style="population")
        
        plt.xticks(rotation=90)
        ax.invert_xaxis()

        plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        plt.title(col_name)
        plt.xlabel("Days before diagnosis")
        plt.ylabel("Cumulative percentage of population")

        plt.legend(bbox_to_anchor=[1.05,1], loc=2)

# COMMAND ----------

df_with_without_xray_activity[(df_with_without_xray_activity["dataset"]=="overall") & (df_with_without_xray_activity["days_between_activity_diagnosis"]==0)].sort_values(by=["col_name", "population"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## overall

# COMMAND ----------

time_to_diagnosis_red_flag = "time_diagnosis_from_first_reported_red_flag_in_last_" + str(history_days) + "_days"

df_no_red_flags = df_pd_patient_flags[df_pd_patient_flags[time_to_diagnosis_red_flag].isnull()]
df_red_flags = df_pd_patient_flags[df_pd_patient_flags[time_to_diagnosis_red_flag].notnull()]

time_to_diagnosis_amber_flag = "time_diagnosis_from_first_reported_amber_flag_in_last_" + str(history_days) + "_days"

df_no_amber_flags = df_pd_patient_flags[df_pd_patient_flags[time_to_diagnosis_amber_flag].isnull()]
df_amber_flags = df_pd_patient_flags[df_pd_patient_flags[time_to_diagnosis_amber_flag].notnull()]

# COMMAND ----------

pct_with_red_flag = df_red_flags.shape[0]/df_pd_patient_flags.shape[0]*100
print(f"{pct_with_red_flag:.1f}% of patients have a red flag in the last {str(history_days)} days")

pct_with_amber_flag = df_amber_flags.shape[0]/df_pd_patient_flags.shape[0]*100
print(f"{pct_with_amber_flag:.1f}% of patients have an amber flag in the last {str(history_days)} days")

# COMMAND ----------

pct_with_red_flag_diagnosis_day = df_red_flags[df_red_flags[time_to_diagnosis_red_flag] == 0].shape[0]/df_pd_patient_flags.shape[0]*100
print(f"{pct_with_red_flag_diagnosis_day:.1f}% of patients have red flag reported on diagnosis date")

pct_with_red_flag_diagnosis_day_of_those_who_reported = df_red_flags[df_red_flags[time_to_diagnosis_red_flag] == 0].shape[0]/df_red_flags.shape[0]*100
print(f"{pct_with_red_flag_diagnosis_day_of_those_who_reported:.1f}% of patients have red flag reported on diagnosis date, of those who reported a red flag symptom")


# COMMAND ----------

df_red_flags[time_to_diagnosis_red_flag].describe().to_frame().transpose()

# COMMAND ----------

df_red_flags[df_red_flags[time_to_diagnosis_red_flag]!=0][time_to_diagnosis_red_flag].describe().to_frame().transpose()



# COMMAND ----------

plt.figure(figsize=(3,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_red_flags, y=time_to_diagnosis_red_flag)
plt.ylabel("Days Between Red Flag Symptom and Diagnosis")
plt.title("Distribution of Time Between Earliest Reported Red Flag Symptom and Diagnosis")
plt.show()

# COMMAND ----------

df_amber_flags[time_to_diagnosis_amber_flag].describe().to_frame().transpose()

# COMMAND ----------

plt.figure(figsize=(3,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_amber_flags, y=time_to_diagnosis_amber_flag)
plt.ylabel("Days Between Amber Flag Symptom and Diagnosis")
plt.title("Distribution of Time Between Earliest Reported Amber Flag Symptom and Diagnosis")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## stage

# COMMAND ----------

# overall breakdown of stage
patient_pathways_utils.count_and_pct(df_pd_patient_flags, "tumour_stage_group")

# COMMAND ----------

# MAGIC %md
# MAGIC ### red flag

# COMMAND ----------

time_diagnosis_value = {}
distribution_metrics = {}

for tumour_stage in df_red_flags["tumour_stage_group"].unique():

    plot_df = df_red_flags[df_red_flags["tumour_stage_group"]==tumour_stage]

    if tumour_stage == "early":
        text="Stage 1 or 2"

    elif tumour_stage == "late":
        text="Stage 3 or 4"

    else:
        text = "unknown"

    patient_pathways_utils.plot_histplot(plot_df=plot_df, 
                col_to_plot=time_to_diagnosis_red_flag,
                title=f"Distribution of Time Between Symptom and Diagnosis\n({text}, Red Flag)",
                xlabel="Days Between Symptom and Diagnosis",
                ylabel="Density",
                figsize=(10,6),
                bins=52,
                kde=True,
                stat="density",
                multiple="layer",
                common_norm=False)

    time_diagnosis_value[tumour_stage] = list(plot_df[time_to_diagnosis_red_flag])
    distribution_metrics[tumour_stage] = patient_pathways_utils.calculate_distribution_metrics(plot_df, time_to_diagnosis_red_flag)
       
    print(distribution_metrics[tumour_stage])

# COMMAND ----------

list_values = list(time_diagnosis_value.values())

if len(list_values) > 1:
    print(anderson_ksamp(list_values))
    print("\n")
    print(kruskal(*list_values))

    print("\n", "----------------------", "\n")

else:
    print("Insufficient data")


# COMMAND ----------

plt.figure(figsize=(10,6))
sns.set(font_scale = 1.5, style="white")

sns.boxplot(data=df_red_flags, x="tumour_stage_group", y=time_to_diagnosis_red_flag, order = ["early", "late", "unknown"], palette = pal_stage,)
plt.ylabel("Days Between Symptom and Diagnosis")
plt.title("Distribution of Time Between Symptom and Diagnosis\n(Red Flag)")
plt.show()

# COMMAND ----------

df_descriptives_time_diagnosis_red_flag_by_stage = df_red_flags.groupby("tumour_stage_group")[time_to_diagnosis_red_flag].describe().reset_index()
df_distribution_time_diagnosis_red_flag_by_stage = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "tumour_stage_group"})

df_summary_time_diagnosis_red_flag_by_stage = df_descriptives_time_diagnosis_red_flag_by_stage.merge(df_distribution_time_diagnosis_red_flag_by_stage[["tumour_stage_group", "range", "skewness", "kurtosis"]],
                                                                                                          on="tumour_stage_group",
                                                                                                          how="inner")

# COMMAND ----------

df_summary_time_diagnosis_red_flag_by_stage

# COMMAND ----------

# MAGIC %md
# MAGIC ### amber flag

# COMMAND ----------

# stage broken down by whether or not an amber flag symptom was reported in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags,
                                     col = "tumour_stage_group",
                                     groupby_col="flag_any_amber_flag_in_last_" + str(history_days) + "_days")

# COMMAND ----------

time_diagnosis_value = {}
distribution_metrics = {}

for tumour_stage in df_amber_flags["tumour_stage_group"].unique():

    plot_df = df_amber_flags[df_amber_flags["tumour_stage_group"]==tumour_stage]

    if tumour_stage == "early":
        text="Stage 1 or 2"

    elif tumour_stage == "late":
        text="Stage 3 or 4"

    else:
        text = "unknown"

    patient_pathways_utils.plot_histplot(plot_df=plot_df, 
                col_to_plot=time_to_diagnosis_amber_flag,
                title=f"Distribution of Time Between Symptom and Diagnosis\n({text}, Amber Flag)",
                xlabel="Days Between Symptom and Diagnosis",
                ylabel="Density",
                figsize=(10,6),
                bins=52,
                kde=True,
                stat="density",
                multiple="layer",
                common_norm=False)


    time_diagnosis_value[tumour_stage] = list(plot_df[time_to_diagnosis_amber_flag])
    distribution_metrics[tumour_stage] = patient_pathways_utils.calculate_distribution_metrics(plot_df, time_to_diagnosis_amber_flag)
       
    print(distribution_metrics[tumour_stage])

# COMMAND ----------

list_values = list(time_diagnosis_value.values())

if len(list_values) > 1:
    print(anderson_ksamp(list_values))
    print("\n")
    print(kruskal(*list_values))

    print("\n", "----------------------", "\n")

else:
    print("Insufficient data")

# COMMAND ----------

if df_amber_flags.shape[0] > 0:

    plt.figure(figsize=(10,6))
    sns.set(font_scale = 1.5, style="white")

    sns.boxplot(data=df_amber_flags, x="tumour_stage_group", y=time_to_diagnosis_amber_flag)
    plt.ylabel("Days Between Symptom and Diagnosis")
    plt.title("Distribution of Time Between Symptom and Diagnosis\n(amber Flag)")
    plt.show()

    df_descriptives_time_diagnosis_amber_flag_by_stage = df_amber_flags.groupby("tumour_stage_group")[time_to_diagnosis_amber_flag].describe().reset_index()
    df_distribution_time_diagnosis_amber_flag_by_stage = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "tumour_stage_group"})

    df_summary_time_diagnosis_amber_flag_by_stage = df_descriptives_time_diagnosis_amber_flag_by_stage.merge(df_distribution_time_diagnosis_amber_flag_by_stage[["tumour_stage_group", "range", "skewness", "kurtosis"]],
                                                                                                            on="tumour_stage_group",
                                                                                                            how="inner")
    
    display(df_summary_time_diagnosis_amber_flag_by_stage)

else:
    print("Insufficient data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## route to diagnosis

# COMMAND ----------

# overall breakdown of stage
patient_pathways_utils.count_and_pct(df_pd_patient_flags, "route_earliest")

# COMMAND ----------

# MAGIC %md
# MAGIC ### red flag

# COMMAND ----------

# stage broken down by whether or not a red flag symptom was reported in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest", groupby_col="flag_any_red_flag_in_last_" + str(history_days) + "_days")

# COMMAND ----------

plt.figure(figsize=(6,6))
sns.set(font_scale = 1, style="white")

sns.barplot(hue="route_earliest",
            y="percentage",
            x = "flag_any_red_flag_in_last_" + str(history_days) + "_days",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "route_earliest", groupby_col="flag_any_red_flag_in_last_" + str(history_days) + "_days"))

plt.legend(bbox_to_anchor = [1.05,1], loc = 2)



# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="route_earliest",
            y="percentage",
            hue = "flag_any_red_flag_in_last_" + str(history_days) + "_days",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "flag_any_red_flag_in_last_" + str(history_days) + "_days", groupby_col="route_earliest"))

plt.legend(bbox_to_anchor = [1.05,1], loc = 2)

# COMMAND ----------

patient_pathways_utils.plot_histplot(plot_df=df_red_flags, 
              col_to_plot=time_to_diagnosis_red_flag,
              title="Distribution of Time Between Symptom and Diagnosis\n(Red Flag)",
              xlabel="Days Between Symptom and Diagnosis",
              ylabel="Density",
              figsize=(10,6),
              hue="route_earliest",
              bins=52,
              kde=True,
              stat="density",
              multiple="layer",
              common_norm=False)

# COMMAND ----------

time_diagnosis_value = {}
distribution_metrics = {}

for route_earliest in df_red_flags["route_earliest"].unique():

    plot_df = df_red_flags[df_red_flags["route_earliest"]==route_earliest]


    patient_pathways_utils.plot_histplot(plot_df=plot_df, 
                col_to_plot=time_to_diagnosis_red_flag,
                title=f"Distribution of Time Between Symptom and Diagnosis\n({text}, Red Flag)",
                xlabel="Days Between Symptom and Diagnosis",
                ylabel="Density",
                figsize=(10,6),
                bins=52,
                kde=True,
                stat="density",
                multiple="layer",
                common_norm=False)

    time_diagnosis_value[route_earliest] = list(plot_df[time_to_diagnosis_red_flag])
    distribution_metrics[route_earliest] = patient_pathways_utils.calculate_distribution_metrics(plot_df, time_to_diagnosis_red_flag)
       
    print(distribution_metrics[route_earliest])

# COMMAND ----------

list_values = list(time_diagnosis_value.values())

if len(list_values) > 1:
    print(anderson_ksamp(list_values))
    print("\n")
    print(kruskal(*list_values))

    print("\n", "----------------------", "\n")

else:
    print("Insufficient data")


# COMMAND ----------

plt.figure(figsize=(10,6))
sns.set(font_scale = 1.5, style="white")

sns.boxplot(data=df_red_flags, x="route_earliest", order = ["USC", "GP referral", "Emergency presentation", "Screening", "Unknown"], y=time_to_diagnosis_red_flag)
plt.ylabel("Days Between Symptom and Diagnosis")
plt.title("Distribution of Time Between Symptom and Diagnosis\n(Red Flag)")
plt.xticks(rotation=90);

# COMMAND ----------

df_descriptives_time_diagnosis_red_flag_by_stage = df_red_flags.groupby("route_earliest")[time_to_diagnosis_red_flag].describe().reset_index()
df_distribution_time_diagnosis_red_flag_by_stage = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "route_earliest"})

df_summary_time_diagnosis_red_flag_by_stage = df_descriptives_time_diagnosis_red_flag_by_stage.merge(df_distribution_time_diagnosis_red_flag_by_stage[["route_earliest", "range", "skewness", "kurtosis"]],
                                                                                                          on="route_earliest",
                                                                                                          how="inner")

df_summary_time_diagnosis_red_flag_by_stage

# COMMAND ----------

# MAGIC %md
# MAGIC ### amber flag

# COMMAND ----------

# stage broken down by whether or not an amber flag symptom was reported in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags,
                                     col = "route_earliest",
                                     groupby_col="flag_any_amber_flag_in_last_" + str(history_days) + "_days")

# COMMAND ----------

patient_pathways_utils.plot_histplot(plot_df=df_amber_flags, 
              col_to_plot=time_to_diagnosis_amber_flag,
              title="Distribution of Time Between Symptom and Diagnosis\n(amber Flag)",
              xlabel="Days Between Symptom and Diagnosis",
              ylabel="Density",
              figsize=(10,6),
              hue="route_earliest",
              bins=52,
              kde=True,
              stat="density",
              multiple="layer",
              common_norm=False)

# COMMAND ----------

time_diagnosis_value = {}
distribution_metrics = {}

for tumour_stage in df_amber_flags["route_earliest"].unique():

    plot_df = df_amber_flags[df_amber_flags["route_earliest"]==tumour_stage]

    if tumour_stage == "early":
        text="Stage 1 or 2"

    elif tumour_stage == "late":
        text="Stage 3 or 4"

    else:
        text = "unknown"

    patient_pathways_utils.plot_histplot(plot_df=plot_df, 
                col_to_plot=time_to_diagnosis_amber_flag,
                title=f"Distribution of Time Between Symptom and Diagnosis\n({text}, Amber Flag)",
                xlabel="Days Between Symptom and Diagnosis",
                ylabel="Density",
                figsize=(10,6),
                bins=52,
                kde=True,
                stat="density",
                multiple="layer",
                common_norm=False)


    time_diagnosis_value[tumour_stage] = list(plot_df[time_to_diagnosis_amber_flag])
    distribution_metrics[tumour_stage] = patient_pathways_utils.calculate_distribution_metrics(plot_df, time_to_diagnosis_amber_flag)
       
    print(distribution_metrics[tumour_stage])

# COMMAND ----------

list_values = list(time_diagnosis_value.values())

if len(list_values) > 1:
    print(anderson_ksamp(list_values))
    print("\n")
    print(kruskal(*list_values))

    print("\n", "----------------------", "\n")

else:
    print("Insufficient data")

# COMMAND ----------

if df_amber_flags.shape[0] > 0:

    plt.figure(figsize=(10,6))
    sns.set(font_scale = 1.5, style="white")

    sns.boxplot(data=df_amber_flags, x="route_earliest", y=time_to_diagnosis_amber_flag)
    plt.ylabel("Days Between Symptom and Diagnosis")
    plt.title("Distribution of Time Between Symptom and Diagnosis\n(amber Flag)")
    plt.xticks(rotation = 90)
    plt.show()

    df_descriptives_time_diagnosis_amber_flag_by_stage = df_amber_flags.groupby("route_earliest")[time_to_diagnosis_amber_flag].describe().reset_index()
    df_distribution_time_diagnosis_amber_flag_by_stage = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "route_earliest"})

    df_summary_time_diagnosis_amber_flag_by_stage = df_descriptives_time_diagnosis_amber_flag_by_stage.merge(df_distribution_time_diagnosis_amber_flag_by_stage[["route_earliest", "range", "skewness", "kurtosis"]],
                                                                                                            on="route_earliest",
                                                                                                            how="inner")
    
    df_summary_time_diagnosis_amber_flag_by_stage

else:
    print("Insufficient data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## deprivation

# COMMAND ----------

# overall breakdown
patient_pathways_utils.count_and_pct(df_pd_patient_flags, "imd_decile_group")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Red flag

# COMMAND ----------

# stage broken down by whether or not a red flag symptom was reported in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "imd_decile_group", groupby_col="flag_any_red_flag_in_last_" + str(history_days) + "_days")

# COMMAND ----------

patient_pathways_utils.plot_histplot(plot_df=df_red_flags, 
              col_to_plot=time_to_diagnosis_red_flag,
              title="Distribution of Time Between Symptom and Diagnosis\n(Red Flag)",
              xlabel="Days Between Symptom and Diagnosis",
              ylabel="Density",
              figsize=(10,6),
              hue="imd_decile_group",
              bins=52,
              kde=True,
              stat="density",
              multiple="layer",
              common_norm=False)

# COMMAND ----------

distribution_metrics = {}
time_diagnosis_value = {}

for imd_decile_group in df_red_flags["imd_decile_group"].unique():

    plot_df = df_red_flags[df_red_flags["imd_decile_group"]==imd_decile_group]

    patient_pathways_utils.plot_histplot(plot_df=plot_df, 
                col_to_plot=time_to_diagnosis_red_flag,
                title=f"Distribution of Time Between Symptom and Diagnosis\n({imd_decile_group}, Red Flag)",
                xlabel="Days Between Symptom and Diagnosis",
                ylabel="Density",
                figsize=(10,6),
                bins=52,
                kde=True,
                stat="density",
                multiple="layer",
                common_norm=False)

    time_diagnosis_value[imd_decile_group] = list(plot_df[time_to_diagnosis_red_flag])
    distribution_metrics[imd_decile_group] = patient_pathways_utils.calculate_distribution_metrics(plot_df, time_to_diagnosis_red_flag)
       
    print(distribution_metrics[imd_decile_group])

# COMMAND ----------

list_values = list(time_diagnosis_value.values())

if len(list_values) > 1:
    print(anderson_ksamp(list_values))
    print("\n")
    print(kruskal(*list_values))

    print("\n", "----------------------", "\n")

else:
    print("Insufficient data")


# COMMAND ----------

plt.figure(figsize=(10,6))
sns.set(font_scale = 1.5, style="white")

sns.boxplot(data=df_red_flags, x="imd_decile_group", y=time_to_diagnosis_red_flag, order=["decile_1_to_3", "decile_4_to_7", "decile_8_to_10", "unknown"])
plt.ylabel("Days Between Symptom and Diagnosis")
plt.title("Distribution of Time Between Symptom and Diagnosis\n(Red Flag)")
plt.show()

# COMMAND ----------

df_descriptives_time_diagnosis_red_flag_by_imd_decile_group = df_red_flags.groupby("imd_decile_group")[time_to_diagnosis_red_flag].describe().reset_index()
df_distribution_time_diagnosis_red_flag_by_imd_decile_group = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "imd_decile_group"})

df_summary_time_diagnosis_red_flag_by_imd_decile_group = df_descriptives_time_diagnosis_red_flag_by_imd_decile_group.merge(df_distribution_time_diagnosis_red_flag_by_imd_decile_group[["imd_decile_group", "range", "skewness", "kurtosis"]],
                                                                                                          on="imd_decile_group",
                                                                                                          how="inner")

df_summary_time_diagnosis_red_flag_by_imd_decile_group

# COMMAND ----------

# MAGIC %md
# MAGIC ### Amber flag

# COMMAND ----------

# stage broken down by whether or not a red flag symptom was reported in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags,
                                     col = "imd_decile_group",
                                     groupby_col="flag_any_amber_flag_in_last_" + str(history_days) + "_days")

# COMMAND ----------

patient_pathways_utils.plot_histplot(plot_df=df_amber_flags, 
              col_to_plot=time_to_diagnosis_amber_flag,
              title="Distribution of Time Between Symptom and Diagnosis\n(amber Flag)",
              xlabel="Days Between Symptom and Diagnosis",
              ylabel="Density",
              figsize=(10,6),
              hue="imd_decile_group",
              bins=52,
              kde=True,
              stat="density",
              multiple="layer",
              common_norm=False)

# COMMAND ----------

distribution_metrics = {}
time_diagnosis_value = {}

for imd_decile_group in df_amber_flags["imd_decile_group"].unique():

    plot_df = df_amber_flags[df_amber_flags["imd_decile_group"]==imd_decile_group]

    patient_pathways_utils.plot_histplot(plot_df=plot_df, 
                col_to_plot=time_to_diagnosis_amber_flag,
                title=f"Distribution of Time Between Symptom and Diagnosis\n({imd_decile_group}, amber Flag)",
                xlabel="Days Between Symptom and Diagnosis",
                ylabel="Density",
                figsize=(10,6),
                bins=52,
                kde=True,
                stat="density",
                multiple="layer",
                common_norm=False)
    
    distribution_metrics[imd_decile_group] = patient_pathways_utils.calculate_distribution_metrics(plot_df, time_to_diagnosis_amber_flag)
    time_diagnosis_value[imd_decile_group] = list(plot_df[time_to_diagnosis_amber_flag])
       
    print(distribution_metrics[imd_decile_group])

# COMMAND ----------

list_values = list(time_diagnosis_value.values())

if len(list_values) > 1:
    print(anderson_ksamp(list_values))
    print("\n")
    print(kruskal(*list_values))

    print("\n", "----------------------", "\n")

else:
    print("Insufficient data")


# COMMAND ----------

if df_amber_flags.shape[0]>0:

    plt.figure(figsize=(10,6))
    sns.set(font_scale = 1.5, style="white")

    sns.boxplot(data=df_amber_flags, x="imd_decile_group", y=time_to_diagnosis_amber_flag, order=["decile_1_to_3", "decile_4_to_7", "decile_8_to_10", "unknown"])
    plt.ylabel("Days Between Symptom and Diagnosis")
    plt.title("Distribution of Time Between Symptom and Diagnosis\n(amber Flag)")
    plt.show()

    df_descriptives_time_diagnosis_amber_flag_by_imd_decile_group = df_amber_flags.groupby("imd_decile_group")[time_to_diagnosis_amber_flag].describe().reset_index()
    df_distribution_time_diagnosis_amber_flag_by_imd_decile_group = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "imd_decile_group"})

    df_summary_time_diagnosis_amber_flag_by_imd_decile_group = df_descriptives_time_diagnosis_amber_flag_by_imd_decile_group.merge(df_distribution_time_diagnosis_amber_flag_by_imd_decile_group[["imd_decile_group", "range", "skewness", "kurtosis"]],
                                                                                                            on="imd_decile_group",
                                                                                                            how="inner")
    
    df_summary_time_diagnosis_amber_flag_by_imd_decile_group

else:
    print("Insufficient data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Age

# COMMAND ----------

# stage broken down by whether or not a red flag symptom was reported in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "flag_any_red_flag_in_last_" + str(history_days) + "_days", groupby_col="age_10yr_band")



# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band",
            y="count",
            hue = "flag_any_red_flag_in_last_" + str(history_days) + "_days",
            data = patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "flag_any_red_flag_in_last_" + str(history_days) + "_days", groupby_col="age_10yr_band")
)

plt.xticks(rotation=90);


# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="age_10yr_band",
            y="percentage",
            hue = "flag_any_red_flag_in_last_" + str(history_days) + "_days",
            data = patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "flag_any_red_flag_in_last_" + str(history_days) + "_days", groupby_col="age_10yr_band")
)

plt.legend(bbox_to_anchor=[1.05,1], loc = 2, title = "flag_any_red_flag_in_last_" + str(history_days) + "_days")

plt.xticks(rotation=90);

# COMMAND ----------

plt.figure(figsize=(10,6))
sns.set(font_scale = 1.5, style="white")

age_bands_order = list(df_red_flags["age_10yr_band"].unique())
age_bands_order.sort()

sns.boxplot(data=df_red_flags, x="age_10yr_band", y=time_to_diagnosis_red_flag, order = age_bands_order)
plt.ylabel("Days Between Symptom and Diagnosis")
plt.title("Distribution of Time Between Symptom and Diagnosis\n(Red Flag)")
plt.xticks(rotation=90);


# COMMAND ----------

distribution_metrics = {}
time_diagnosis_value = {}

for age_10yr_band in df_red_flags["age_10yr_band"].unique():

    plot_df = df_red_flags[df_red_flags["age_10yr_band"]==age_10yr_band]

    time_diagnosis_value[age_10yr_band] = list(plot_df[time_to_diagnosis_red_flag])
    distribution_metrics[age_10yr_band] = patient_pathways_utils.calculate_distribution_metrics(plot_df, time_to_diagnosis_red_flag)
    

df_descriptives_time_diagnosis_red_flag_by_age_10yr_band = df_red_flags.groupby("age_10yr_band")[time_to_diagnosis_red_flag].describe().reset_index()
df_distribution_time_diagnosis_red_flag_by_age_10yr_band = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "age_10yr_band"})

df_summary_time_diagnosis_red_flag_by_age_10yr_band = df_descriptives_time_diagnosis_red_flag_by_age_10yr_band.merge(df_distribution_time_diagnosis_red_flag_by_age_10yr_band[["age_10yr_band", "range", "skewness", "kurtosis"]],
                                                                                                          on="age_10yr_band",
                                                                                                          how="inner")

df_summary_time_diagnosis_red_flag_by_age_10yr_band

# COMMAND ----------

# MAGIC %md
# MAGIC ## timeline of reporting of red flag symptoms

# COMMAND ----------

# MAGIC %md
# MAGIC ### GP data

# COMMAND ----------

unique_symptom_names = df_gp_red_flags.select("symptom_name").distinct().rdd.flatMap(lambda x: x).collect()
print(unique_symptom_names)

df_symptoms_cumulative_counts = pd.DataFrame()

for symptom in unique_symptom_names:
    df_symptom_first_per_patient_per_day_pd = patient_pathways_utils.cumulative_count_table(df_gp_red_flags.filter(F.col("symptom_name")==symptom), history_days)
    df_symptom_first_per_patient_per_day_pd["symptom"] = symptom
    df_symptom_first_per_patient_per_day_pd["cumulative_percentage"] = 100*df_symptom_first_per_patient_per_day_pd["cumulative_count"]/df_pd_patient_flags.shape[0]
    df_symptoms_cumulative_counts = pd.concat([df_symptoms_cumulative_counts,df_symptom_first_per_patient_per_day_pd], axis =0)

df_symptoms_cumulative_counts

# COMMAND ----------

plt.figure(figsize=(10,8))
sns.set(font_scale = 1, style="white")

ax = sns.lineplot(x="days_between_activity_diagnosis",
                  y="cumulative_percentage",
                  data = df_symptoms_cumulative_counts,
                  hue = "symptom")

plt.xticks(rotation=90)
ax.invert_xaxis()

plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

plt.xlabel("Days before cancer diagnosis")
plt.ylabel("Cumulative percentage of first patient GP report of red flag symptom")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Acute

# COMMAND ----------

df_acute_first_per_patient_per_day_pd = patient_pathways_utils.cumulative_count_table(df_acute_red_flag , history_days)
df_acute_first_per_patient_per_day_pd["cumulative_percentage"] = 100*df_acute_first_per_patient_per_day_pd["cumulative_count"]/df_pd_patient_flags.shape[0]

plt.figure(figsize=(10,8))
sns.set(font_scale = 1, style="white")

ax = sns.lineplot(x="days_between_activity_diagnosis",
                  y="cumulative_percentage",
                  data = df_acute_first_per_patient_per_day_pd)

plt.xticks(rotation=90)
ax.invert_xaxis()

plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

plt.xlabel("Days before cancer diagnosis")
plt.ylabel("Cumulative percentage of first patient acute red flag symptom")

# COMMAND ----------

# MAGIC %md
# MAGIC # Metric 2 - Frequency of reported relevant red/amber flag symptoms in the year before cancer diagnosis (broken down by early/late diagnosis or deprivation and channel where they were reported, A&E, 111, gp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## overall frequency

# COMMAND ----------

df_pd_patient_flags.shape


# COMMAND ----------

df_pd_patient_flags[list_red_flag_count_columns + list_amber_flag_count_columns].describe().transpose()

# COMMAND ----------

plt.figure(figsize=(8,8))
sns.set(font_scale = 1, style="white")

df_pd_patient_flags_long_red_flag = pd.melt(df_pd_patient_flags, id_vars = ["Patient_ID"], value_vars = list_red_flag_count_columns)

sns.barplot(x="variable", y="value", data = df_pd_patient_flags_long_red_flag)
plt.ylabel("Mean number of times red flag symptom reported per patient")

plt.xticks(rotation=90);


# COMMAND ----------

# MAGIC %md
# MAGIC ## stage

# COMMAND ----------

for col in list_red_flag_count_columns + list_amber_flag_count_columns:
    print(col)
    display(df_pd_patient_flags.groupby(["tumour_stage_group"])[col].describe().reset_index())


# COMMAND ----------

plt.figure(figsize=(8,8))
sns.set(font_scale = 1, style="white")

df_pd_patient_flags_long_red_flag = pd.melt(df_pd_patient_flags, id_vars = ["Patient_ID", "tumour_stage_group"], value_vars = list_red_flag_count_columns)

sns.barplot(y="variable", x="value", hue = "tumour_stage_group", data = df_pd_patient_flags_long_red_flag)
plt.xlabel("Mean number of times red flag symptom reported per patient")
plt.ylabel("Source")
plt.legend(bbox_to_anchor=[1.05,1], title = "tumour_stage_group")

# COMMAND ----------

# MAGIC %md
# MAGIC ## deprivation

# COMMAND ----------

for col in list_red_flag_count_columns + list_amber_flag_count_columns:
    print(col)
    display(df_pd_patient_flags.groupby(["imd_decile_group"])[col].describe().reset_index())

# COMMAND ----------

plt.figure(figsize=(8,8))
sns.set(font_scale = 1, style="white")

df_pd_patient_flags_long_red_flag = pd.melt(df_pd_patient_flags, id_vars = ["Patient_ID", "imd_decile_group"], value_vars = list_red_flag_count_columns)

sns.barplot(y="variable", x="value", hue = "imd_decile_group", data = df_pd_patient_flags_long_red_flag)
plt.xlabel("Mean number of times red flag symptom reported per patient")
plt.ylabel("Source")
plt.legend(bbox_to_anchor=[1.05,1], title = "imd_decile_group")

# COMMAND ----------

# MAGIC %md
# MAGIC ## route to diagnosis

# COMMAND ----------

for col in list_red_flag_count_columns + list_amber_flag_count_columns:
    print(col)
    display(df_pd_patient_flags.groupby(["route_earliest"])[col].describe().reset_index())

# COMMAND ----------

plt.figure(figsize=(8,8))
sns.set(font_scale = 1, style="white")

df_pd_patient_flags_long_red_flag = pd.melt(df_pd_patient_flags, id_vars = ["Patient_ID", "route_earliest"], value_vars = list_red_flag_count_columns)

sns.barplot(y="variable", x="value", hue = "route_earliest", data = df_pd_patient_flags_long_red_flag, hue_order = ["USC", "GP referral","Screening", "Emergency presentation"])
plt.xlabel("Mean number of times red flag symptom reported per patient")
plt.ylabel("Source")
plt.legend(bbox_to_anchor=[1.05,1], title = "route_earliest")

# COMMAND ----------

# MAGIC %md
# MAGIC # Any red flag in history

# COMMAND ----------

# stage broken down by whether or not a red flag symptom was reported in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col="flag_any_red_flag_in_last_" + str(history_days) + "_days")

# COMMAND ----------

plt.figure(figsize=(6,6))
sns.set(font_scale = 1, style="white")

sns.barplot(hue="tumour_stage_group",
            y="percentage",
            x = "flag_any_red_flag_in_last_" + str(history_days) + "_days",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col="flag_any_red_flag_in_last_" + str(history_days) + "_days"))

plt.legend(bbox_to_anchor = [1.05,1], loc = 2)



# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = f"flag_any_red_flag_in_last_" + str(history_days) + "_days", groupby_col="tumour_stage_group").pivot_table(index="tumour_stage_group", columns="flag_any_red_flag_in_last_" + str(history_days) + "_days", values=["count", "percentage"]).fillna(0)

# COMMAND ----------

plt.figure(figsize=(6,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="tumour_stage_group",
            y="percentage",
            hue = "flag_any_red_flag_in_last_" + str(history_days) + "_days",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "flag_any_red_flag_in_last_" + str(history_days) + "_days", groupby_col="tumour_stage_group"))

plt.legend(bbox_to_anchor = [1.05,1], loc = 2, title = "flag_any_red_flag_in_last_" + str(history_days) + "_days")



# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = f"flag_any_red_flag_in_last_" + str(history_days) + "_days", groupby_col="route_earliest").pivot_table(index="route_earliest", columns="flag_any_red_flag_in_last_" + str(history_days) + "_days", values=["count", "percentage"]).fillna(0)

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.set(font_scale = 1, style="white")

sns.barplot(x="route_earliest",
            y="percentage",
            hue = "flag_any_red_flag_in_last_" + str(history_days) + "_days",
            data=patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = f"flag_any_red_flag_in_last_" + str(history_days) + "_days", groupby_col="route_earliest"))

plt.legend(bbox_to_anchor = [1.05,1], loc = 2, title = "flag_any_red_flag_in_last_" + str(history_days) + "_days")
plt.xticks(rotation=90);

# COMMAND ----------

df_pd_patient_flags[(df_pd_patient_flags["route_earliest"]=="Emergency presentation") & (df_pd_patient_flags[f"flag_any_red_flag_in_last_" + str(history_days) + "_days"])==1]["time_diagnosis_from_first_reported_red_flag_in_last_" + str(history_days) + "_days"].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # When are Symptoms Reported?

# COMMAND ----------

def plot_cumsum(flag_df, flag_date_col, ax, label, color="blue"):
    flag_df = flag_df.withColumn("days_from_first_symptom_to_diagnosis", F.datediff(F.col("diagnosis_date_earliest"), F.col(flag_date_col))).toPandas()
    total_patients = len(flag_df)
    cumsum = {}
    for i in range(history_days):
        val = len(flag_df[flag_df["days_from_first_symptom_to_diagnosis"] >= i])
        cumsum[i] = 100*val/total_patients

    ax.plot(cumsum.keys(), cumsum.values(), label=label, color=color)
    ax.set_xlabel("Days before diagnosis")
    ax.set_ylabel("cummulative % of patients")
    ax.set_xlim(history_days,0)

# COMMAND ----------

ed_flag_cols = ['date_first_reported_ecds_red_flag_in_last_365_days', 'date_first_reported_ecds_amber_flag_in_last_365_days']
df_patient_flags = df_patient_flags.withColumn("date_first_reported_ecds_flag", F.least(*ed_flag_cols))

amber_and_red_flag_cols = ['date_first_reported_ecds_red_flag_in_last_365_days', 'date_first_reported_111_red_flag_in_last_365_days', 'date_first_reported_gp_red_flag_in_last_365_days', 'date_first_reported_acute_red_flag_in_last_365_days', 'date_first_reported_ecds_amber_flag_in_last_365_days', 'date_first_reported_111_amber_flag_in_last_365_days', 'date_first_reported_gp_amber_flag_in_last_365_days', 'date_first_reported_acute_amber_flag_in_last_365_days']
df_patient_flags = df_patient_flags.withColumn("date_first_reported_any_flag", F.least(*amber_and_red_flag_cols))

gp_flag_cols = ['date_first_reported_gp_red_flag_in_last_365_days', 'date_first_reported_gp_amber_flag_in_last_365_days']
df_patient_flags = df_patient_flags.withColumn("date_first_reported_gp_flag", F.least(*gp_flag_cols))

# COMMAND ----------

fig, ax = plt.subplots(1,1, figsize=(10,5))

for route, color in pal_route.items():
    route_df = df_patient_flags.filter(F.col("route_earliest") == route)
    plot_cumsum(route_df, "date_first_reported_any_flag", ax, label=route, color=color)
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
ax.set_title("Patients Who Reported Any Symptom Flag Before Diagnosis")

# COMMAND ----------

fig, ax = plt.subplots(1,1, figsize=(10,5))

for route, color in pal_route.items():
    route_df = df_patient_flags.filter(F.col("route_earliest") == route)
    plot_cumsum(route_df, "date_first_reported_ecds_flag", ax, label=route, color=color)
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
ax.set_title("Patients Who Reported Any Emergency Flag Before Diagnosis")

# COMMAND ----------

fig, ax = plt.subplots(1,1, figsize=(10,5))

for route, color in pal_route.items():
    route_df = df_patient_flags.filter(F.col("route_earliest") == route)
    plot_cumsum(route_df, "date_first_reported_gp_flag", ax, label=route, color=color)
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
ax.set_title("Patients Who Reported Any GP Flag Before Diagnosis")

# COMMAND ----------

fig, ax = plt.subplots(1,1, figsize=(10,5))

for stage, color in pal_stage.items():
    route_df = df_patient_flags.filter(F.col("tumour_stage_group") == stage)
    plot_cumsum(route_df, "date_first_reported_any_flag", ax, label=stage, color=color)
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
ax.set_title("Patients Who Reported Any NG12 Flag Before Diagnosis")

# COMMAND ----------

fig, ax = plt.subplots(1,1, figsize=(10,5))

for stage, color in pal_stage.items():
    route_df = df_patient_flags.filter(F.col("tumour_stage_group") == stage)
    plot_cumsum(route_df, "date_first_reported_ecds_flag", ax, label=stage, color=color)
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
ax.set_title("Patients Who Reported Any Emergency Flag Before Diagnosis")

# COMMAND ----------

fig, ax = plt.subplots(1,1, figsize=(10,5))

for stage, color in pal_stage.items():
    route_df = df_patient_flags.filter(F.col("tumour_stage_group") == stage)
    plot_cumsum(route_df, "date_first_reported_gp_flag", ax, label=stage, color=color)
plt.legend(bbox_to_anchor=[1.05,1], loc=2)
ax.set_title("Patients Who Reported Any GP Flag Before Diagnosis")

# COMMAND ----------

# MAGIC %md
# MAGIC # Metric 3: Time from referral to cancer diagnosis (broken down by early/late diagnosis or deprivation).
# MAGIC

# COMMAND ----------

df_gp_referred = df_pd_patient_flags[df_pd_patient_flags[f"flag_gp_referral_suspected_in_last_{str(history_days)}_days"]==1]

pct_with_gp_referral = df_gp_referred.shape[0]/df_pd_patient_flags.shape[0]*100
print(f"{pct_with_gp_referral:.1f}% of patients have a gp referral in the last {str(history_days)} days")


# COMMAND ----------

# MAGIC %md
# MAGIC ## overall

# COMMAND ----------

df_gp_referred["time_diagnosis_from_gp_referral_suspected"].describe().to_frame().transpose()

# COMMAND ----------

plt.figure(figsize=(3,6))
sns.set(font_scale = 1, style="white")

sns.boxplot(data=df_gp_referred, y="time_diagnosis_from_gp_referral_suspected")
plt.ylabel("Days Between gp referral and Diagnosis")
plt.title("Distribution of Time Between gp Referral and Diagnosis")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage

# COMMAND ----------

# stage broken down by whether or not there was a gp referral in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "tumour_stage_group", groupby_col=f"flag_gp_referral_suspected_in_last_{str(history_days)}_days")

# COMMAND ----------

time_diagnosis_value = {}
distribution_metrics = {}

for tumour_stage in df_red_flags["tumour_stage_group"].unique():

    plot_df = df_gp_referred[df_gp_referred["tumour_stage_group"]==tumour_stage]

    if tumour_stage == "early":
        text="Stage 1 or 2"

    elif tumour_stage == "late":
        text="Stage 3 or 4"

    else:
        text = "unknown"

    patient_pathways_utils.plot_histplot(plot_df=plot_df, 
                col_to_plot="time_diagnosis_from_gp_referral_suspected",
                title=f"Distribution of Time Between gp referral and Diagnosis\n{text}",
                xlabel="Days Between Referral and Diagnosis",
                ylabel="Density",
                figsize=(10,6),
                bins=52,
                kde=True,
                stat="density",
                multiple="layer",
                common_norm=False)
    
    time_diagnosis_value[tumour_stage] = list(plot_df["time_diagnosis_from_gp_referral_suspected"])
    distribution_metrics[tumour_stage] = patient_pathways_utils.calculate_distribution_metrics(plot_df, "time_diagnosis_from_gp_referral_suspected")
       
    print(distribution_metrics[tumour_stage])

# COMMAND ----------

list_values = list(time_diagnosis_value.values())

print("Early, Late, and Unknown")

if len(list_values) > 1:
    print(anderson_ksamp(list_values))
    print("\n")
    print(kruskal(*list_values))

    print("\n", "----------------------", "\n")

else:
    print("Insufficient data")

print("Only Early and Late")

print(anderson_ksamp([time_diagnosis_value["early"], time_diagnosis_value["late"]]))
print("\n")
print(kruskal(time_diagnosis_value["early"], time_diagnosis_value["late"]))

# COMMAND ----------

plt.figure(figsize=(10,6))
sns.set(font_scale = 1.5, style="white")

sns.boxplot(data=df_gp_referred, x="tumour_stage_group", y="time_diagnosis_from_gp_referral_suspected", order = ["early", "late", "unknown"], palette = pal_stage,)
plt.ylabel("Days Between gp referral and Diagnosis")
plt.title("Distribution of Time Between Referral and Diagnosis")
plt.show()

# COMMAND ----------

df_descriptives_time_diagnosis_referred_by_stage = df_gp_referred.groupby("tumour_stage_group")["time_diagnosis_from_gp_referral_suspected"].describe().reset_index()
df_distribution_time_diagnosis_referred_by_stage = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "tumour_stage_group"})

df_summary_time_diagnosis_referred_by_stage = df_descriptives_time_diagnosis_referred_by_stage.merge(df_distribution_time_diagnosis_referred_by_stage[["tumour_stage_group", "range", "skewness", "kurtosis"]],
                                                                                                          on="tumour_stage_group",
                                                                                                          how="inner")

df_summary_time_diagnosis_referred_by_stage

# COMMAND ----------

# MAGIC %md
# MAGIC ## deprivation

# COMMAND ----------

# stage broken down by whether or not there was a gp referral in the last 365 days
patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = "imd_decile_group", groupby_col=f"flag_gp_referral_suspected_in_last_{str(history_days)}_days")

# COMMAND ----------

distribution_metrics = {}

for imd_decile in df_red_flags["imd_decile_group"].unique():

    plot_df = df_gp_referred[df_gp_referred["imd_decile_group"]==imd_decile]

    patient_pathways_utils.plot_histplot(plot_df=plot_df, 
                col_to_plot="time_diagnosis_from_gp_referral_suspected",
                title=f"Distribution of Time Between gp referral and Diagnosis\n{imd_decile_group}",
                xlabel="Days Between Referral and Diagnosis",
                ylabel="Density",
                figsize=(10,6),
                bins=52,
                kde=True,
                stat="density",
                multiple="layer",
                common_norm=False)
    
    time_diagnosis_value[imd_decile] = list(plot_df["time_diagnosis_from_gp_referral_suspected"])
    distribution_metrics[imd_decile] = patient_pathways_utils.calculate_distribution_metrics(plot_df, "time_diagnosis_from_gp_referral_suspected")
       
    print(distribution_metrics[imd_decile])

# COMMAND ----------

list_values = list(time_diagnosis_value.values())

if len(list_values) > 1:
    print(anderson_ksamp(list_values))
    print("\n")
    print(kruskal(*list_values))

    print("\n", "----------------------", "\n")

else:
    print("Insufficient data")


# COMMAND ----------

plt.figure(figsize=(10,6))
sns.set(font_scale = 1.5, style="white")

sns.boxplot(data=df_gp_referred, x="imd_decile_group", y="time_diagnosis_from_gp_referral_suspected", order=["decile_1_to_3", "decile_4_to_7", "decile_8_to_10", "unknown"])
plt.ylabel("Days Between gp referral and Diagnosis")
plt.title("Distribution of Time Between Referral and Diagnosis")
plt.show()

# COMMAND ----------

df_descriptives_time_diagnosis_referred_by_imd_decile = df_gp_referred.groupby("imd_decile_group")["time_diagnosis_from_gp_referral_suspected"].describe().reset_index()
df_distribution_time_diagnosis_referred_by_imd_decile = pd.DataFrame.from_dict(distribution_metrics, orient='index').reset_index().rename(columns = {"index": "imd_decile_group"})

df_summary_time_diagnosis_referred_by_imd_decile = df_descriptives_time_diagnosis_referred_by_imd_decile.merge(
                                                                                                            df_distribution_time_diagnosis_referred_by_imd_decile[["imd_decile_group", "range", "skewness", "kurtosis"]],
                                                                                                            on="imd_decile_group",
                                                                                                            how="inner")

df_summary_time_diagnosis_referred_by_imd_decile

# COMMAND ----------

# MAGIC %md
# MAGIC ## comparison of gp referral from SNOMED codes with referral from cancer registry route to diagnosis

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags, col = f"flag_gp_referral_suspected_in_last_{str(history_days)}_days", groupby_col="route_earliest").pivot_table(index="route_earliest", columns="flag_gp_referral_suspected_in_last_365_days", values = ["count", "percentage"]).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## timeline of referral

# COMMAND ----------

df_referral_first_per_patient_per_day_pd = patient_pathways_utils.cumulative_count_table(df_referral, history_days)
df_referral_first_per_patient_per_day_pd["cumulative_percentage"] = 100*df_referral_first_per_patient_per_day_pd["cumulative_count"]/df_pd_patient_flags.shape[0]
df_referral_first_per_patient_per_day_pd["activity"] = "GP referral"

plt.figure(figsize=(10,8))
sns.set(font_scale = 1, style="white")

ax = sns.lineplot(x="days_between_activity_diagnosis",
                  y="cumulative_percentage",
                  data = df_referral_first_per_patient_per_day_pd)

plt.xticks(rotation=90)
ax.invert_xaxis()

plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

plt.xlabel("Days before cancer diagnosis")
plt.ylabel("Cumulative percentage of first patient GP referral")

# COMMAND ----------

df_combine_symptoms_with_referral = pd.concat([df_symptoms_cumulative_counts, df_referral_first_per_patient_per_day_pd], axis=0)
df_combine_symptoms_with_referral["category"] = np.where(df_combine_symptoms_with_referral["activity"].notnull(),
                                                         df_combine_symptoms_with_referral["activity"],
                                                         df_combine_symptoms_with_referral["symptom"]
                                                         )

# COMMAND ----------

plt.figure(figsize=(10,8))
sns.set(font_scale = 1, style="white")

ax = sns.lineplot(x="days_between_activity_diagnosis",
                  y="cumulative_percentage",
                  data = df_combine_symptoms_with_referral,
                  hue="category")

plt.xticks(rotation=90)
ax.invert_xaxis()

plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

plt.xlabel("Days before cancer diagnosis")
plt.ylabel("Cumulative percentage of first patient activity")

# COMMAND ----------

# MAGIC %md
# MAGIC # Metric 7 - Distribution of stage at diagnosis for people who attended/not attended screening (in the year before diagnosis). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## using route to diagnosis

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags[df_pd_patient_flags["route_earliest"]=="Screening"], "tumour_stage_group")

# COMMAND ----------

patient_pathways_utils.count_and_pct(df_pd_patient_flags[df_pd_patient_flags["route_earliest"]!="Screening"], "tumour_stage_group")

# COMMAND ----------

# not diagnosed via screening, and in same age range

patient_pathways_utils.count_and_pct(df_pd_patient_flags[(df_pd_patient_flags["route_earliest"]!="Screening") & 
                                                        (df_pd_patient_flags["Age"]>=50) & (df_pd_patient_flags["Age"]<71)], "tumour_stage_group")

# COMMAND ----------

# MAGIC %md
# MAGIC ## using gp SNOMED codelist

# COMMAND ----------

# recorded as attending screening
patient_pathways_utils.count_and_pct(df_pd_patient_flags[df_pd_patient_flags["flag_attended_screening"]==1], "tumour_stage_group")

# COMMAND ----------

# did not have screening 
patient_pathways_utils.count_and_pct(df_pd_patient_flags[df_pd_patient_flags["flag_attended_screening"]==0], "tumour_stage_group")

# COMMAND ----------

# Those indicated as DNA or declined, and did not get diagnosed via screening
patient_pathways_utils.count_and_pct(df_pd_patient_flags[(df_pd_patient_flags["flag_did_not_attend_screening"]==1) & 
                                    (df_pd_patient_flags["route_earliest"] != "Screening")], "tumour_stage_group")

# COMMAND ----------

# Those indicated as DNA or declined, and did not get diagnosed via screening
patient_pathways_utils.count_and_pct(df_pd_patient_flags[(df_pd_patient_flags["flag_did_not_attend_screening"]==1) & 
                                    (df_pd_patient_flags["route_earliest"] != "Screening")], "imd_decile_group")

# COMMAND ----------

# MAGIC %md
# MAGIC ## degree of overlap between gp SNOMED coding of screening, and route to diagnosis

# COMMAND ----------

df_pd_patient_flags["flag_attended_screening"].value_counts()

# COMMAND ----------

df_pd_patient_flags["flag_abnormal_screening"].value_counts()

# COMMAND ----------

df = df_pd_patient_flags.copy()
df["flag_attended_screening"] = df["flag_attended_screening"].fillna(0)

patient_pathways_utils.count_and_pct(df, col = "flag_attended_screening", groupby_col="route_earliest").pivot_table(index="route_earliest", columns="flag_attended_screening", values = ["count", "percentage"]).fillna(0)

# COMMAND ----------

df = df_pd_patient_flags.copy()
df["flag_abnormal_screening"] = df["flag_abnormal_screening"].fillna(0)

patient_pathways_utils.count_and_pct(df,
                                     col = "flag_abnormal_screening",
                                     groupby_col="route_earliest").pivot_table(index="route_earliest", columns="flag_abnormal_screening", values = ["count", "percentage"]).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualise single patient journey

# COMMAND ----------

# select an ID to visualise patient journey for

visualise_patient_joruney = True
if visualise_patient_joruney == True:
    selected_id = df_pd_patient_flags.iloc[5]['Patient_ID']


    df_patient_activity = df_all_activity.filter((F.col("Patient_ID") == selected_id) & (F.col("days_between_activity_diagnosis")<=365) &  (F.col("days_between_activity_diagnosis")>=0)).select(["Patient_ID", "date", "dataset", "days_between_activity_diagnosis", "description"]).toPandas()

    df_patient_activity["description"] = df_patient_activity["description"].fillna("")

    # Group terms by event type and day
    x_var = "date"

    df_grouped = df_patient_activity.groupby(["dataset", x_var])["description"].apply(lambda x: "<br>".join(x)).reset_index()

    # Assign y-position to each Dataset for stacking
    dataset_order = {etype: i for i, etype in enumerate(sorted(df_grouped["dataset"].unique(), reverse=True))}
    df_grouped["y"] = df_grouped["dataset"].map(dataset_order)

    # Create scatter plot
    fig = px.scatter(
        df_grouped,
        x=x_var,
        y="y",
        hover_data={"description": True, x_var: True, "dataset": False},
        text=None
    )

    # Format dots
    fig.update_traces(marker=dict(size=12, color='blue'))

    # Clean layout for stacked timeline
    fig.update_layout(
        title="Stacked Event Timeline",
        xaxis_title=x_var,
        yaxis=dict(
            tickmode="array",
            tickvals=list(dataset_order.values()),
            ticktext=list(dataset_order.keys()),
            title="Event Type"
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=False,
        height=400 + 60 * len(dataset_order)  # dynamic height
    )

    fig.show()

# COMMAND ----------

visualise_patient_joruney = False
if visualise_patient_joruney == True:

    list_ids = list(df_pd_patient_flags[(df_pd_patient_flags["route_earliest"]=="Emergency presentation") & (df_pd_patient_flags["number_of_times_gp_red_flag_in_last_365_days"]>0)]['Patient_ID'])

    for selected_id in list_ids: 

        df_patient_activity = df_all_activity.filter((F.col("Patient_ID") == selected_id) & (F.col("days_between_activity_diagnosis")<=365) &  (F.col("days_between_activity_diagnosis")>=0)).select(["Patient_ID", "date", "dataset", "days_between_activity_diagnosis", "description"]).toPandas()

        df_patient_activity["description"] = df_patient_activity["description"].fillna("")

        # Group terms by event type and day
        x_var = "date"

        df_grouped = df_patient_activity.groupby(["dataset", x_var])["description"].apply(lambda x: "<br>".join(x)).reset_index()

        # Assign y-position to each Dataset for stacking
        dataset_order = {etype: i for i, etype in enumerate(sorted(df_grouped["dataset"].unique(), reverse=True))}
        df_grouped["y"] = df_grouped["dataset"].map(dataset_order)

        # Create scatter plot
        fig = px.scatter(
            df_grouped,
            x=x_var,
            y="y",
            hover_data={"description": True, x_var: True, "dataset": False},
            text=None
        )

        # Format dots
        fig.update_traces(marker=dict(size=12, color='blue'))

        # Clean layout for stacked timeline
        fig.update_layout(
            title="Stacked Event Timeline",
            xaxis_title=x_var,
            yaxis=dict(
                tickmode="array",
                tickvals=list(dataset_order.values()),
                ticktext=list(dataset_order.keys()),
                title="Event Type"
            ),
            hoverlabel=dict(bgcolor="white", font_size=12),
            showlegend=False,
            height=400 + 60 * len(dataset_order)  # dynamic height
        )

        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Sankey Diagram

# COMMAND ----------

def generate_sankey_diagram(df, group1, group2, group3):
    # Prepare flows between levels
    sources_targets = []

    # First level: flag -> route
    level1 = df.groupby([group1, group2]).size().reset_index(name='count')
    for _, row in level1.iterrows():
        sources_targets.append((row[group1], row[group2], row['count']))

    # Second level: route -> stage
    level2 = df.groupby([group2, group3]).size().reset_index(name='count')
    for _, row in level2.iterrows():
        sources_targets.append((row[group2], row[group3], row['count']))

    # Create list of unique labels
    first_layer = list(df[group1].sort_values().unique())
    second_layer = list(df[group2].sort_values().unique())
    third_layer = list(df[group3].sort_values().unique())

    labels = first_layer + second_layer + third_layer

    # Map labels to indices
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Build source, target, and value lists
    sources = [label_to_index[src] for src, tgt, val in sources_targets]
    targets = [label_to_index[tgt] for src, tgt, val in sources_targets]
    values = [val for src, tgt, val in sources_targets]

    # Calculate percentages for link labels
    total = sum(values)
    percentages = [f"{(val/total)*100:.1f}%" for val in values]

    # Color coding for tumour stages
    stage_colors = {
        'early': 'green',
        'late': 'red',
        'unknown': 'gray'
    }

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=percentages
        )
    )])

    fig.update_layout(title_text="Patient Flow Sankey Diagram with Percentages and Stage Colors", font_size=12)

    fig.show()


# COMMAND ----------

df_pd_patient_flags['flag_any_red_flag_in_last_365_days_text'] = df_pd_patient_flags['flag_any_red_flag_in_last_365_days'].map({0: 'No Red Flag', 1: 'Red Flag'})

generate_sankey_diagram(df = df_pd_patient_flags,
                        group1 = "flag_any_red_flag_in_last_365_days_text",
                        group2 = "route_earliest",
                        group3 = "tumour_stage_group")

# COMMAND ----------

generate_sankey_diagram(df = df_pd_patient_flags,
                        group1 = "age_10yr_band",
                        group2 = "route_earliest",
                        group3 = "tumour_stage_group")

# COMMAND ----------

generate_sankey_diagram(df = df_pd_patient_flags,
                        group1 = "Population_Segment",
                        group2 = "route_earliest",
                        group3 = "tumour_stage_group")

# COMMAND ----------

generate_sankey_diagram(df = df_pd_patient_flags[df_pd_patient_flags["tumour_stage_group"]=="late"],
                        group1 = "age_10yr_band",
                        group2 = "flag_any_red_flag_in_last_365_days_text",
                        group3 = "route_earliest")

# COMMAND ----------

generate_sankey_diagram(df = df_pd_patient_flags[df_pd_patient_flags["time_diagnosis_from_first_reported_red_flag_in_last_365_days"]>=30],
                        group1 = "age_10yr_band",
                        group2 = "route_earliest",
                        group3 = "tumour_stage_group")

# COMMAND ----------

generate_sankey_diagram(df = df_pd_patient_flags[df_pd_patient_flags["time_diagnosis_from_first_reported_red_flag_in_last_365_days"]<30],
                        group1 = "age_10yr_band",
                        group2 = "route_earliest",
                        group3 = "tumour_stage_group")
