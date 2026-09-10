# Contents

# Overview #

This code utilises linked NHS patient-level records to determine the events which preceded a patient's diagnosis with lung cancer. Our data includes information from a range of different healthcare settings. We examined patient pathways to try and determine possible earlier points for intervention, as well as to identify points where delays are most prevelant in pathways. 

We were especially interested in how actual patient journeys compared to the referral pathways layed out in the [NG12 guidelines](https://www.cancerresearchuk.org/health-professional/diagnosis/primary-care/suspected-cancer-referral-guidelines/nice-ng12) for cancer diagnosis. These define a set of symptom and demographic criteria which if met by a patient should lead to subsequent investiagtions such as chest x-rays. 

# Data

This project used healthcare data provided by the [Leeds Data Model](https://www.healthandcareleeds.org/healthy-leeds-plan/4-population-health-infrastructure/) (LDM). LDM is a data platform run by West Yorkshire ICB, which provides access to pseudonymised health records for patients registered within Leeds.

The patient-level data used in this project to construct care pathways includes the following. 

- GP appointments and individual events
- GP medication prescriptions
- A&E Attendances
- Inpatient and outpatient secondary care interactions
- Civil Deaths Registry
- National Cancer Registry
- 111 and 999 calls
- Patient demogrpahics such as age and smoking status

We link these data sources together, using a patient pseudo ID, to construct a medical history for each individual patient, allowing us to see their healthcare interactions preceding lung cancer diagnosis.

# Methodology

## Cancer Patients

Using the National Cancer Registry we identified patients who met the following three criteria.

1. Recieved a lung cancer diagnosis between May 2022 and January 2025
2. Had no previous cancer diagnoses of any kind
3. Were registered to a Leeds GP for at least the full year before their cancer diagnosis

Using this criteria we selected 1233 patients whose pathways we examined.

## Control Patients

We also select a control cohort with which to compare the medical histories of the cancer patients. For every cancer patient we find a corresponding patient who was not diagnosed with cancer in the same timeframe. Patients are matched on the following charateristics.

- Age
- Gender
- Smoking Status
- GP Practice (where possible)

We then use this control cohort as a baseline to compare our cancer patients. 

## Symptom and Medication Coding

Symptoms and medictions relevant to lung cancer were taken from the NG12 guidelines. For each symptom we developed code lists containing Snomed, ICD10, procedure codes and 111 DX codes which mapped onto them. These lists can found in the `code_lists` folder. 

## Metrics

Using the combined patient histories we calculate certain metrics to determine the effectiveness of diagnostic pathways. For example we are interested in the number of days between a patient first meeting the NG12 criteria for chest imaging, and the time when they actually got one. Metrics such as these allow us to see how well NG12 guidelines appear to be adhered too, and where possible bottlenecks in the health system might exist.

# Code Structure

The key files in the code are described below.

```bash

├───notebooks
│   ├───patient_events
│       │
│       ├───1_patient_events_construction_lung
│       ├───2_patient_journeys_construction_lung     
│       ├───3_patient_events_construction_lung_control
|       ├───4_patient_journeys_construction_lung_control
|       ├───5_metrics_journey_lung
|       ├───ppv_calculation_for_symptoms     
└───src
    └───code_lists
    └───cancer_late
        |
        ├───__init__.py
        ├───config.py
        ├───config_pathways.py
        ├───patient_pathways_utils.py
        ├───processing.py
        ├───utils.py

```

The primary purposes of the 5 patient events notebooks is as follows.

1. Contructs an events table for patients within the cancer cohort, combining records from the sources listed above. 
2. Uses the combined events table created by the previous notebook to generate tables capturing important information in a patient's journey in the year prior to their lung cancer diagnosis. Important information captured includes the time prior to diagnosis when NG12 symptoms were first reported, the date when chest imaging was performed on a patient prior to diagnosis, and the date when medications such as inhalers or antibiotics were prescribed to a patient.
3. Using the criteria mentioned above a demographically matched control cohort is constructed with which to compare the cancer cohort to. The primary output is a combined events table as in the first notebook.
4. Analagous to notebook 2, except journey tables are constructed for the control cohort. 
5. Notebook to create visualistions and calculate primary metrics eg. average days from NG12 symptom to lung cancer diagnosis. Split into sections which can be run independently.

The final PPV notebook was written to investigate the positive predictive value (PPV) of the symptoms described in the NG12 guidelines. We also used this notebok to check the PPV of potentially novel combinations fo symptoms and patient prescriptions.