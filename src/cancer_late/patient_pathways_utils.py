import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from pyspark.sql import functions as F
import pandas as pd
from pyspark.sql.window import Window
from typing import List
import numpy as np

def count_and_first_date_of_event(df_event_table, history_days, name_of_event):
    """
    For a given event table, returns a summary DataFrame with the count of events and the first date of event occurrence
    for each patient within the specified history window.

    Args:
        df_event_table (DataFrame): Spark DataFrame containing event records with columns 'Patient_ID', 'days_between_activity_diagnosis', and 'date'.
        history_days (int): Number of days to look back from the diagnosis date.
        name_of_event (str): Name of the event for labeling output columns.

    Returns:
        DataFrame: Spark DataFrame with columns:
            - Patient_ID
            - number_of_times_{name_of_event}_in_last_{history_days}_days
            - date_first_reported_{name_of_event}_in_last_{history_days}_days
    """
    df_event_table = df_event_table.filter(F.col("days_between_activity_diagnosis")<=history_days)

    df_event_patient_summary = (
        df_event_table.groupBy("Patient_ID")
        .agg(
            F.count("*").alias("number_of_times_" + name_of_event + "_in_last_" + str(history_days) + "_days"),
            F.min(F.col("date")).alias("date_first_reported_" + name_of_event + "_in_last_" + str(history_days) + "_days")
        )
    )

    return df_event_patient_summary

def create_flags_summary(df_mapped_events, history_days, dataset, col_name, name_of_event):
    """
    Generates a summary of patient events for a given flag type, including overall and per-category counts and first event dates.

    Parameters
    ----------
    df_mapped_events : DataFrame
        Spark DataFrame containing mapped event records with patient and event details.
    history_days : int
        Number of days to look back for event history.
    dataset : str
        Name of the dataset or flag type (e.g., 'ecds', 'gp', '111').
    col_name : str
        Column name representing the event category (e.g., diagnosis or symptom).
    name_of_event : str
        Name to assign to the overall event summary.

    Returns
    -------
    DataFrame
        Patient-level summary DataFrame with counts and first event dates for the overall flag and each category.
    """
    # remove duplicates
    df_mapped_events = df_mapped_events.dropDuplicates(["Patient_ID", "Attendance_Date" ,col_name])

    df_mapped_events = df_mapped_events.filter(F.col("days_between_activity_diagnosis")<=history_days)
    
    display(df_mapped_events.groupby(col_name).count())

    df_mapped_events_patient_summary = count_and_first_date_of_event(df_event_table=df_mapped_events,
                                                                     history_days = history_days,
                                                                     name_of_event=name_of_event)

    # count per diagnosis
    list_categories = [row[0] for row in df_mapped_events.select(col_name).distinct().collect()]

    for category in list_categories:
        df_category_summary = count_and_first_date_of_event(df_event_table=df_mapped_events.filter(F.col(col_name)==category),
                                                            history_days = history_days,
                                                            name_of_event= dataset +"_" + category)
        
        df_mapped_events_patient_summary = df_mapped_events_patient_summary.join(df_category_summary, 
                                                                                         on = "Patient_ID", 
                                                                                         how = "left")
    
    return df_mapped_events_patient_summary

def calculate_distribution_metrics(plot_df, col_to_plot):
    """
    Calculates distribution metrics for a specified column in a DataFrame.

    Args:
        plot_df (DataFrame): Input pandas DataFrame.
        col_to_plot (str): Name of the column to analyze.

    Returns:
        dict: Dictionary containing mean, median, range, skewness, and kurtosis of the column.
    """
    skewness = skew(plot_df[col_to_plot].dropna())
    kurt_val = kurtosis(plot_df[col_to_plot].dropna())
    mean_val = plot_df[col_to_plot].mean()
    median_val = plot_df[col_to_plot].median()
    range_val = plot_df[col_to_plot].max() - plot_df[col_to_plot].min()

    return {
        "mean": mean_val,
        "median": median_val,
        "range": range_val,
        "skewness": skewness,
        "kurtosis": kurt_val}

def plot_histplot(plot_df, 
                  col_to_plot,
                  title,
                  xlabel,
                  ylabel,
                  figsize,
                  **kwargs):
    """
    Plots a histogram for a specified column in a DataFrame using seaborn.

    Args:
        plot_df (DataFrame): Input pandas DataFrame.
        col_to_plot (str): Name of the column to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple): Figure size (width, height).
        **kwargs: Additional keyword arguments for seaborn.histplot, such as:
            - hue (str): Variable in data to map plot aspects to different colors.
            - bins (int or sequence): Number of histogram bins or bin edges.
            - kde (bool): Whether to plot a kernel density estimate.
            - stat (str): Aggregate statistic to compute in each bin.
            - color (str): Color for all elements.
            - multiple (str): Approach to drawing multiple elements.
            - common_norm (bool): Whether to normalize across multiple histograms.

    Returns:
        None. Displays the histogram plot.
    """

    fig, ax = plt.subplots(figsize=figsize)

    sns.set(font_scale = 1, style="white")

    sns.histplot(
        data=plot_df,
        x=col_to_plot,
        hue = kwargs.get("hue"),
        bins=kwargs.get("bins"),
        kde=kwargs.get("kde"),
        stat=kwargs.get("stat"),
        color=kwargs.get("color"),
        multiple=kwargs.get("multiple"), 
        common_norm=kwargs.get("common_norm"),
        ax=ax
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2)
    plt.show()

def count_and_pct(df, col, groupby_col=None):
    """
    Calculates the count and percentage of unique values in a specified column, optionally grouped by another column.

    Args:
        df (DataFrame or pandas.DataFrame): Input DataFrame.
        col (str): Name of the column for which to calculate counts and percentages.
        groupby_col (str, optional): Column to group by before calculating counts and percentages. Defaults to None.

    Returns:
        DataFrame: DataFrame with columns for the value, count, and percentage. If groupby_col is specified, includes groupby_col in the output.
    """

    if groupby_col:
        counts = df.groupby(groupby_col)[col].value_counts().to_frame()
        counts.columns = ["count"]
        percs = df.groupby(groupby_col)[col].value_counts(normalize=True).mul(100).round(1).to_frame()
        percs.columns = ["percentage"]
        df_count_pct = counts.merge(percs, on = [groupby_col, col]).reset_index()
        df_count_pct = df_count_pct.sort_values([groupby_col, col])
        
    else:
        counts = df[col].value_counts()
        percs = df[col].value_counts(normalize=True).mul(100).round(1)
        df_count_pct = pd.concat([counts,percs], axis=1, keys=['count', 'percentage']).reset_index().rename(columns = {'index':col})
        df_count_pct = df_count_pct.sort_values(col)
        
    return df_count_pct

def cumulative_count_table(df_activity_table, history_days):
    """
    Generates a cumulative count table of first activity per patient within a specified history window.

    Parameters
    ----------
    df_activity_table : pyspark.sql.DataFrame
        Input Spark DataFrame containing patient activity records. Must include columns:
        'Patient_ID', 'days_between_activity_diagnosis', 'Term', 'symptom_name', 'grouping', 'date', 'source'.
    history_days : int
        The maximum number of days between activity and diagnosis to include.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - 'days_between_activity_diagnosis': Days between activity and diagnosis (descending from history_days to 0).
        - 'count': Number of patients with first activity at each day.
        - 'cumulative_count': Cumulative count of patients from each day to 0.
    """
    window_spec = Window.partitionBy("Patient_ID").orderBy("date")
    df_activity_table_first_per_patient = df_activity_table.filter(F.col("days_between_activity_diagnosis")<=history_days).withColumn("row_num", F.row_number().over(window_spec)).filter(F.col("row_num") == 1).drop("row_num")

    df_activity_table_first_per_patient_pd = df_activity_table_first_per_patient.select(["Patient_ID", "days_between_activity_diagnosis"]).toPandas()

    df_days_between_activity_diagnosis = pd.DataFrame({'days_between_activity_diagnosis': range(0, history_days + 1)})

    df_activity_table_first_per_patient_per_day_pd = df_activity_table_first_per_patient_pd.groupby("days_between_activity_diagnosis").agg({"Patient_ID": "count"}).reset_index().rename(columns = {"Patient_ID": "count"})

    df_activity_table_first_per_patient_per_day_pd = df_days_between_activity_diagnosis.merge(df_activity_table_first_per_patient_per_day_pd, on ="days_between_activity_diagnosis", how="left").fillna(0)

    df_activity_table_first_per_patient_per_day_pd = df_activity_table_first_per_patient_per_day_pd.sort_values("days_between_activity_diagnosis", ascending=False).reset_index(drop=True)

    df_activity_table_first_per_patient_per_day_pd["cumulative_count"] = df_activity_table_first_per_patient_per_day_pd["count"].cumsum()

    return df_activity_table_first_per_patient_per_day_pd

def create_base_table_from_lists(list1: List, name1: str, list2: List, name2: str, list3: List = None, name3: str =None):
    """
    Creates a base pandas DataFrame representing the Cartesian product of two or three lists, with specified column names.

    Args:
        list1 (list or array-like): First list of values.
        name1 (str): Column name for the first list.
        list2 (list or array-like): Second list of values.
        name2 (str): Column name for the second list.
        list3 (list or array-like, optional): Third list of values. Defaults to None.
        name3 (str, optional): Column name for the third list. Required if list3 is provided.

    Returns:
        pandas.DataFrame: DataFrame containing all combinations of the provided lists as columns.
    """
    df_base_1 = pd.DataFrame(list1, columns = [name1])
    df_base_2 = pd.DataFrame(list2, columns = [name2])

    df_base_table = pd.merge(df_base_1,df_base_2, how="cross" )

    if list3 is not None:
        df_base_3 = pd.DataFrame(list3, columns = [name3])
        df_base_table = pd.merge(df_base_table,df_base_3, how="cross" )

    return df_base_table

def create_long_event_table_from_columns(df_pd_patient_flags,
                                       date_columns_to_process,
                                       id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                       diagnosis_date_col = "diagnosis_date_earliest",
                                       history_days = 365):
    """
    Converts wide-format patient event date columns into a long-format DataFrame, calculating days between each event and diagnosis.

    Args:
        df_pd_patient_flags (pandas.DataFrame): Patient-level DataFrame containing event date columns and identifiers.
        date_columns_to_process (list): List of column names representing event dates to process.
        id_vars (list, optional): Columns to use as identifier variables in the melt operation. Defaults to ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"].
        diagnosis_date_col (str, optional): Name of the column containing the diagnosis date. Defaults to "diagnosis_date_earliest".
        history_days (int, optional): Number of days before diagnosis to include (not used in this function, but kept for interface consistency).

    Returns:
        pandas.DataFrame: Long-format DataFrame with columns:
            - id_vars (identifier columns)
            - 'col_name': Name of the event date column
            - 'earliest_date_of_symptom': Date of the event
            - 'days_between_activity_diagnosis': Days between event and diagnosis date
    """
    subset_cols = []

    for col in date_columns_to_process:
        if col in df_pd_patient_flags.columns:
            subset_cols.append(col)

    df_dates_symptoms_ng12 = pd.melt(df_pd_patient_flags,
                                    id_vars = id_vars,
                                    value_vars= subset_cols,
                                    var_name="col_name",
                                    value_name="earliest_date_of_symptom"
                                    ).dropna(axis=0)

    df_dates_symptoms_ng12[diagnosis_date_col] = pd.to_datetime(df_dates_symptoms_ng12[diagnosis_date_col])
    df_dates_symptoms_ng12['earliest_date_of_symptom'] = pd.to_datetime(df_dates_symptoms_ng12['earliest_date_of_symptom'])

    df_dates_symptoms_ng12["days_between_activity_diagnosis"] = (df_dates_symptoms_ng12[diagnosis_date_col] - df_dates_symptoms_ng12['earliest_date_of_symptom']).dt.days

    return df_dates_symptoms_ng12

def create_cumulative_count_percentage_from_date_col(df_pd_patient_flags,
                                       date_columns_to_process,
                                       id_vars = ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"],
                                       diagnosis_date_col = "diagnosis_date_earliest",
                                       history_days = 365):
    """
    Calculates cumulative counts and percentages of patients with events (e.g., symptoms or flags) occurring within a specified number of days before diagnosis, for multiple date columns.

    Args:
        df_pd_patient_flags (pandas.DataFrame): Patient-level DataFrame containing event date columns and patient identifiers.
        date_columns_to_process (list): List of column names in df_pd_patient_flags representing event dates to process.
        id_vars (list, optional): List of columns to use as identifier variables in the melt operation. Defaults to ["Patient_ID", "tumour_stage_group", "diagnosis_date_earliest"].
        diagnosis_date_col (str, optional): Name of the column containing the diagnosis date. Defaults to "diagnosis_date_earliest".
        history_days (int, optional): Number of days before diagnosis to include in the analysis. Defaults to 365.

    Returns:
        pandas.DataFrame: DataFrame with columns:
            - 'days_between_activity_diagnosis': Days between event and diagnosis (descending from history_days to 0).
            - 'col_name': Name of the event date column.
            - 'count': Number of patients with first event at each day for each event type.
            - 'cumsum_count': Cumulative count of patients for each event type.
            - 'cumulative_percentage': Cumulative percentage of patients for each event type.
    """

    df_dates_symptoms_ng12 = create_long_event_table_from_columns(df_pd_patient_flags,
                                       date_columns_to_process,
                                       id_vars,
                                       diagnosis_date_col,
                                       history_days)

    df_dates_symptoms_ng12_count = df_dates_symptoms_ng12[(df_dates_symptoms_ng12["days_between_activity_diagnosis"]<=history_days) & 
                                                        (df_dates_symptoms_ng12["days_between_activity_diagnosis"]>=0)].groupby(["days_between_activity_diagnosis", "col_name"])["Patient_ID"].count().to_frame().reset_index()

    df_dates_symptoms_ng12_count.rename(columns = {"Patient_ID": "count"}, inplace=True)

    df_base_table = create_base_table_from_lists(list1 = np.linspace(0, history_days, history_days+1),
                                               name1 = "days_between_activity_diagnosis",
                                               list2 = date_columns_to_process,
                                               name2 = "col_name")

    df_dates_symptoms_ng12_count = df_base_table.merge(df_dates_symptoms_ng12_count, on = ["days_between_activity_diagnosis", "col_name"], how="left").fillna(0)

    df_dates_symptoms_ng12_count = df_dates_symptoms_ng12_count.sort_values(["days_between_activity_diagnosis", "col_name"], ascending = False)

    df_dates_symptoms_ng12_count["cumsum_count"] = df_dates_symptoms_ng12_count.groupby("col_name")["count"].cumsum()
    df_dates_symptoms_ng12_count["cumulative_percentage"] = 100*df_dates_symptoms_ng12_count["cumsum_count"]/df_pd_patient_flags.shape[0]

    return df_dates_symptoms_ng12_count

def compare_codes_in_activity(df_all_activity_before_diagnosis,
                              dataset,
                              total_num_unique_patients,
                              column_for_activity = "description",
                              history_days = 365,
                              ):
    """
    Compare the percentage of patients with each activity code in the first half versus the second half of the specified history window before diagnosis.

    Args:
        df_all_activity_before_diagnosis: Spark DataFrame containing activity data before diagnosis.
        dataset: Name of the dataset to filter (e.g., "gp_events").
        column_for_activity: Column name representing the activity code/description.
        history_days: Number of days in the history window before diagnosis.

    Returns:
        Spark DataFrame with, for each code in column_for_activity:
            - unique_ids_first_half: Number of unique patients in the first half of the window.
            - percentage_of_cases_first_half: Percentage of patients in the first half.
            - unique_ids_second_half: Number of unique patients in the second half of the window.
            - percentage_of_cases_second_half: Percentage of patients in the second half.
            - difference_in_pct: Difference in percentage between second and first half.
            - ratio_second_half_to_first_half: Ratio of unique patients in the second half to the first half.
    """

    # find % of patients with a code in first 6 months, compared to latest 6 months 
    df_number_unique_patients_per_code_first_half = df_all_activity_before_diagnosis.filter((F.col("days_between_activity_diagnosis")<=history_days) & 
                                                                                            (F.col("days_between_activity_diagnosis")>=history_days/2) &
                                                                                            (F.col("dataset")==dataset)).groupby(column_for_activity).agg(F.countDistinct("Patient_ID").alias("unique_ids_first_half"))

    df_number_unique_patients_per_code_first_half = df_number_unique_patients_per_code_first_half.withColumn("percentage_of_cases_first_half", 100*F.col("unique_ids_first_half")/total_num_unique_patients)

    df_number_unique_patients_per_code_first_half = df_number_unique_patients_per_code_first_half.filter(F.col(column_for_activity).isNotNull())

    # find % of patients with a code in first 6 months, compared to latest 6 months 
    df_number_unique_patients_per_code_second_half = df_all_activity_before_diagnosis.filter((F.col("days_between_activity_diagnosis")<history_days/2) & 
                                                                                            (F.col("dataset")==dataset)).groupby(column_for_activity).agg(F.countDistinct("Patient_ID").alias("unique_ids_second_half"))

    df_number_unique_patients_per_code_second_half = df_number_unique_patients_per_code_second_half.withColumn("percentage_of_cases_second_half", 100*F.col("unique_ids_second_half")/total_num_unique_patients)

    df_number_unique_patients_per_code_second_half = df_number_unique_patients_per_code_second_half.filter(F.col(column_for_activity).isNotNull())

    # compare the two rates

    df_number_unique_patients_per_code_comparison = df_number_unique_patients_per_code_first_half.join(df_number_unique_patients_per_code_second_half, on = column_for_activity, how = "fullouter").fillna(0)
    df_number_unique_patients_per_code_comparison = df_number_unique_patients_per_code_comparison.withColumn("difference_in_pct", F.col("percentage_of_cases_second_half") - F.col("percentage_of_cases_first_half"))
    df_number_unique_patients_per_code_comparison = df_number_unique_patients_per_code_comparison.withColumn("ratio_second_half_to_first_half", F.col("unique_ids_second_half")/F.col("unique_ids_first_half"))

    return df_number_unique_patients_per_code_comparison
