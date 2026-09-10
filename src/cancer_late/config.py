lakeName=""
containerName_bronze=""
containerName_platinum=""

dict_cohort_paths = {
    "2020-03-31": ",
    "2020-06-30": ",
    "2020-09-30": ",
    "2020-12-31": ",
    "2021-03-31": ",
    "2021-06-30": ",
    "2021-09-30": ",
    "2021-12-31": ",
    "2022-03-31": ",
    "2022-06-30": ",
    "2022-09-30": ",
    "2022-12-31": ",
    "2023-03-31": ",
    "2023-06-30": ",
    "2023-09-30": ",
    "2023-12-31": ",
    "2024-03-31": ",
    "2024-06-30": ",
    "2024-12-31": ",
    "2025-03-31": "
    }

# File paths to import from Bronze container
filePath_acute = ""
filePath_sus=""
filePath_gp_all=""
filePath_gp_events_data_extended=""
filePath_emis = ""
filePath_s1 = ""
filePath_deaths=""
filePath_cancer_registration_registry = ""
filePath_cancer_registration_rapid = ""
filePath_icd10 = ""
filePath_SCT_Concept_Descriptions = ""
filePath_111 = ""
filePath_999 = ""
filePath_OoH = ""
filePath_ecds = ["",
                 "",
                 "",
                 "",
                 "",
                 "",
                 "",
                 ]
filePath_gp_meds = ""

# File paths of reference datasets in platinum container
filePath_snomed = ""
filePath_snomed_flags = ""
filePath_all_cancer_snomed = ""
filePath_111_mapping = ""
filePath_master_patient_table = ""
filePath_cancer_flags_next_52_104_260_weeks = ""
filePath_cancer_registry_earliest_latest_by_cancer_group = ""
filePath_acute_sus_diagnosis_group_flags = ""
filePath_all_patient_diagnoses_primary_secondary = ""
filePath_primary_care_group_flags = ""
filePath_111_symptom_flags = ""
filePath_snomed_hierarchy_code_level = ""
filePath_snomed_hierarchy_group_level = ""
filePath_snomed_cluster_code_level = ""
filePath_multicohort_patient_table = ""
filePath_multicohort_cancer_flags_next_52_104_260_weeks = ""
filePath_multicohort_cancer_registry_earliest_latest_by_cancer_group = ""
filePath_multicohort_acute_sus_diagnosis_group_flags = ""
filePath_multicohort_primary_care_group_flags = ""
filePath_multicohort_folder = ""
filePath_multicohort_with_flags = ""