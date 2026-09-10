# list of filepaths for symptoms
chest_pain = {"name":"chest_pain",
              "path":"/code_lists/lung/amber_flag_gp/chest_pain/opensafely-symptoms-chest-pain-unexplained-0d0d5f59.csv"}

cough = {"name":"cough",
         "path":"/code_lists/lung/amber_flag_gp/cough/opensafely-symptoms-cough-55dfd783.csv"}

fatigue = {"name":"fatigue",
            "path":"/code_lists/lung/amber_flag_gp/fatigue/opensafely-symptoms-fatigue-0e9ac677.csv"}

weight_loss = {"name":"weight_loss",
               "path":"/code_lists/lung/amber_flag_gp/weight_loss/Structure-weight-loss-5067859d.csv"}

appetite_loss = {"name":"appetite_loss",
               "path":"/code_lists/lung/amber_flag_gp/appetite_loss/Structure-appetite-loss-37581268.csv"}

shortness_breath = {"name":"shortness_breath",
                    "path":"/code_lists/lung/amber_flag_gp/shortness_breath/opensafely-symptoms-breathlessness-new-5aee78ee.csv"}

haemoptysis = {"name":"haemoptysis",
               "path":"/code_lists/lung/red_flag_gp/haemoptysis/Structure-unexplained-haemoptysis-07ebebf4.csv"}

supraclavicular_lymphadenopathy = {"name":"supraclavicular_lymphadenopathy",
                                   "path":"/code_lists/lung/red_flag_gp/supraclavicular_lymphadenopathy/Structure-supraclavicular-lymphadenopathy-6c29b8c8.csv"}

cervical_lymphadenopathy = {"name":"cervical_lymphadenopathy",
                            "path":"/code_lists/lung/red_flag_gp/cervical_lymphadenopathy/Structure-cervical-lymphadenopathy-531a4593.csv"}

thrombocytosis = {"name":"thrombocytosis",
                  "path":"/code_lists/lung/red_flag_gp/thrombocytosis/Structure-thrombocytosis-20fb5f29.csv"}

finger_clubbing = {"name":"finger_clubbing",
                  "path":"/code_lists/lung/red_flag_gp/finger_clubbing/Structure-finger-clubbing-05392bfe.csv"}

chest_infection = {"name": "chest_infection",
                   "path": "/code_lists/lung/red_flag_gp/chest_infection/sx_ci_ab.csv"}

# 111 files are in format cancer_site_flags_111 e.g. breast_flags_111

# cancer referral
general_suspected_referral = "/code_lists/lung/cancer_referral/general_cancer_referral_suspected-03568894.csv"
lung_cancer_suspected_referral = "/code_lists/lung/cancer_referral/lung_cancer_referral-187b1101.csv"

# chest x-ray
chest_xray_snomed = "/code_lists/lung/chest_xray_snomed/chest-x-ray-6e582f81.csv"

# lung cancer meds data
oral_antibiotics_snomed = "/code_lists/lung/medication_flags/oral_antibiotics.csv"
inhalers_snomed = "/code_lists/lung/medication_flags/all_inhalers.csv"

# codes for proxy smoking flag
smoking_cessation_snomed = "/code_lists/lung/smoking_flag/smoking_cessation.csv"

# severe mental illness
smi_snomed = "/code_lists/smi/nhs_data_model_severe_mental_illness_icd10.csv"
smi_icd10 = "/code_lists/smi/qcovid-has_severe_mental_illness-351410b2.csv"
smi_look_back_years = 2

# GP deprivation 
gp_deprivation = "/code_lists/gp_deprivation/GP_IMD_interpolated.csv"

# cancer site file mapping

cancer_site_mappings = {
                            "Lung": {
                                       "sources_of_flags" : ["gp", "111", "ecds", "acute"],
                                       "red_flag_gp": [chest_infection, finger_clubbing, supraclavicular_lymphadenopathy, cervical_lymphadenopathy, thrombocytosis, haemoptysis], 
                                       "amber_flag_gp": [cough, fatigue, shortness_breath, chest_pain, weight_loss, appetite_loss],
                                       "flags_111": "/code_lists/lung/flags_111/lung_flags_111.csv",
                                       "flags_ecds": "/code_lists/lung/flags_ecds/lung_ecds_chief_complaint.csv",
                                       "icd10_red_flag_codes": "/code_lists/lung/icd10_red_flag_codes/lung_icd10_red_flag_codes.csv",
                                       "icd10_amber_flag_codes": "/code_lists/lung/icd10_amber_flag_codes/lung_icd10_amber_flag_codes.csv",
                                       "cancer_referral": [general_suspected_referral,lung_cancer_suspected_referral],
                                       "screening": True,
                                       "screening_attended": "/code_lists/lung/screening_attended/attended_lung_health_check-47d4657b.csv",
                                       "screening_did_not_attend": "/code_lists/lung/screening_did_not_attend/did_not_attend_lung_health_check-021e5dc2.csv",
                                       "screening_icd10_attended": "Z122",
                                       "screening_icd10_abnormal": "R91",
                                       "unexplained_symptoms_NICE_guidelines": ["cough", "fatigue", "shortness_breath", "chest_pain", "weight_loss", "appetite_loss"],
                                       "critical_symptoms_NICE_guidelines": ["chest_infection", "finger_clubbing", "supraclavicular_lymphadenopathy", "cervical_lymphadenopathy", "thrombocytosis", "haemoptysis"],
                                       "medication_flags": ["oral_antibiotic", "inhaler"],
                                       "chest_xray_snomed" : chest_xray_snomed,
                                       "abnormal_xray_icd10" : "R91",
                                       "medication": True,
                                       "medication_code_tables": {"oral_antibiotic": oral_antibiotics_snomed, "inhaler": inhalers_snomed},
                                       "copd_icd10": "/code_lists/lung/copd_icd10/bristol-copd-0031c5f4.csv",
                                       "copd_snomed": "/code_lists/lung/copd_snomed/qcovid-has_copd-7adcea65.csv",
                                       "copd_look_back_years": 30,
                                       "run_version": "20260414",
                            }
}