from __future__ import annotations

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
APP_DIR = REPO_DIR
APP_DATA_DIR = APP_DIR / "data"
PROJECT_DIR = REPO_DIR
CFDNA_ROOT_DIR = REPO_DIR
DATA_DIR = REPO_DIR / "data"
MODEL_METADATA_PATH = APP_DATA_DIR / "model_metadata.json"
SCHEMA_METADATA_PATH = APP_DATA_DIR / "schema_metadata.json"

BASELINE_CSV_FILENAME = "demo_survival_supplementary_no30_death_only_v5.csv"
CHECKPOINT_FALLBACK = (
    PROJECT_DIR
    / "results"
    / "death_only"
    / "optuna_v5_maskhead_nosite_nopeakdsa_noweights_noaucddpct"
    / "retrain_checkpoints"
    / "death_only"
    / (
        "multitask_deephit_v1_4_cfdna_optuna_death_only_"
        "v5_maskhead_nosite_nopeakdsa_noweights_noaucddpct-20260418v1_"
        "trial506_all_data_retrain_best_model.pt"
    )
)
SHAP_CACHE_DIR = PROJECT_DIR / "results" / "kernel_shap_cache_trial506_noaucddpct"

CAT_EMBEDDING_DIM = 4
EMBEDDING_MIN_CLASSES = 3
MC_SAMPLES_DEFAULT = 128
MC_INTERVAL_ALPHA = 0.10
SHAP_BACKGROUND_SIZE = 64
SHAP_NSAMPLES = 500
USE_SAVED_SHAP_DEFAULT = True
FEV1_TARGET_MONTH = 12.0
MORTALITY_TARGET_YEARS = 2.0

CATEGORY_COLUMNS = [
    "final_TRANSPLANT_TYPE",
    "SEX",
    "ETHNICITY",
    "NATIVE_LUNG_DISEASE_Coded",
    "RACE",
    "DONOR_CMV_SEROLOGY",
    "DONOR_GENDER",
    "DONOR_ETHNICITY",
    "DONOR_CAUSE_OF_DEATH",
    "DONOR_HX_DIABETES",
    "DONOR_HX_CANCER",
    "DONOR_HX_HYPERTENSION",
    "DONOR_HX_CIG",
    "DONOR_HX_ALCOHOL",
    "DONOR_HX_DRUG",
    "RECIPIENT_CMV_SEROLOGY",
    "DIABETES",
    "VENOUS_THROMBOEMBOLISM",
    "PULM_EMBOLISM",
    "SUPPLEMENTAL_OXYGEN",
    "VENTILATOR",
    "ECMO",
    "PGD_Grade_3",
    "DSA_present",
    "AMR_present",
    "HLA_MISMATCH_LVL",
    "BLOOD_TYPE_RECIPIENT",
    "SMOKING_HISTORY_RECIPIENT",
    "Race_mismatch",
    "Blood_type_mismatch",
]

EXCLUDED_PREDICTOR_COLUMNS = {
    "SITE",
    "Peak_DSA_strength",
    "final_WEIGHT_KG_TX",
    "DONOR_WT",
    "AUC_ddcfDNA_PCT_1_30",
}

TARGET_SPECS = {
    "mortality_2y": {"label": "2-year mortality risk", "unit": "%"},
    "fev1_1y": {"label": "1-year FEV1", "unit": "FEV1"},
    "severe_ACR": {"label": "Severe ACR probability", "unit": "%"},
    "ever_clinical_AMR": {"label": "Clinical AMR probability", "unit": "%"},
    "BLAD": {"label": "BLAD probability", "unit": "%"},
}

FEATURE_LABEL_MAP = {
    "final_HEIGHTCM_AT_TX": "Recipient height at transplant",
    "final_WEIGHT_KG_TX": "Recipient weight at transplant",
    "AGE": "Recipient age at transplant",
    "LAS_AT_TRANSPLANT": "LAS at transplant",
    "BMI_AT_TX": "Recipient BMI at transplant",
    "DONOR_AGE": "Donor age",
    "DONOR_KDPI": "Donor KDPI",
    "DONOR_WT": "Donor weight",
    "DONOR_HT": "Donor height",
    "DONOR_BMI": "Donor BMI",
    "TOTAL_ISCHEMIC_TIME_HOURS_HR_L": "Total ischemic time",
    "AUC_ddcfDNA_PCT_1_30": "AUC of dd-cfDNA (%) over 30 days",
    "AUC_ddcfDNA_Copies_1_30": "AUC of dd-cfDNA (copies/mL) over 30 days",
    "AUC_rdcfDNA_Copies_1_30": "AUC of rd-cfDNA (copies/mL) over 30 days",
    "log2_rdcfDNA_.Copies.mL.": "Log2 rd-cfDNA (copies/mL)",
    "final_TRANSPLANT_TYPE": "Transplant type",
    "SEX": "Recipient sex",
    "ETHNICITY": "Recipient ethnicity",
    "DONOR_CMV_SEROLOGY": "Donor CMV serostatus",
    "DONOR_GENDER": "Donor sex",
    "DONOR_HX_DIABETES": "Donor history of diabetes",
    "DONOR_HX_CANCER": "Donor history of cancer",
    "DONOR_HX_HYPERTENSION": "Donor history of hypertension",
    "DONOR_HX_CIG": "Donor smoking history",
    "DONOR_HX_ALCOHOL": "Donor alcohol use history",
    "DONOR_HX_DRUG": "Donor drug use history",
    "RECIPIENT_CMV_SEROLOGY": "Recipient CMV serostatus",
    "DIABETES": "Recipient diabetes",
    "VENOUS_THROMBOEMBOLISM": "History of venous thromboembolism",
    "PULM_EMBOLISM": "History of pulmonary embolism",
    "SUPPLEMENTAL_OXYGEN": "Supplemental oxygen before transplant",
    "VENTILATOR": "Mechanical ventilation before transplant",
    "ECMO": "ECMO before transplant",
    "PGD_Grade_3": "PGD grade 3",
    "DSA_present": "DSA present within 30 days",
    "AMR_present": "AMR within 30 days",
    "SMOKING_HISTORY_RECIPIENT": "Recipient smoking history",
    "Race_mismatch": "Donor-recipient race mismatch",
    "Blood_type_mismatch": "Donor-recipient ABO mismatch",
    "SITE": "Transplant center",
    "NATIVE_LUNG_DISEASE_Coded": "Native lung disease diagnosis",
    "DONOR_ETHNICITY": "Donor ethnicity",
    "DONOR_CAUSE_OF_DEATH": "Donor cause of death",
    "Peak_DSA_strength": "Peak DSA strength within 30 days",
    "HLA_MISMATCH_LVL": "HLA mismatch level",
    "RACE": "Recipient race",
    "BLOOD_TYPE_RECIPIENT": "Recipient blood type",
}

CATEGORY_VALUE_LABEL_MAP = {
    "final_TRANSPLANT_TYPE": {0: "Double", 1: "Single", -1: "Heart Lung or missing"},
    "SITE": {
        0: "INOVA FAIRFAX HOSPITAL",
        1: "JOHNS HOPKINS HOSPITAL",
        2: "UNIVERSITY OF MARYLAND",
        -1: "Missing",
    },
    "SEX": {0: "MALE", 1: "FEMALE", -1: "Missing"},
    "ETHNICITY": {
        0: "NOT LATINO OR HISPANIC (includes UNKNOWN mapped to 0)",
        1: "LATINO OR HISPANIC",
        -1: "Missing",
    },
    "DONOR_CMV_SEROLOGY": {
        0: "Positive",
        1: "Negative",
        -1: "Indeterminate / Not Done / Missing",
    },
    "DONOR_GENDER": {0: "MALE", 1: "FEMALE"},
    "DONOR_ETHNICITY": {0: "Non-hispanic", 1: "Hispanic", 2: "Others", -1: "Missing"},
    "DONOR_HX_HYPERTENSION": {0: "No", 1: "Yes", -1: "Missing"},
    "DONOR_CAUSE_OF_DEATH": {
        0: "Head Trauma",
        1: "Anoxia",
        2: "Cerebrovascular / Stroke",
        3: "Other / Specific",
        -1: "Missing / Unknown",
    },
    "RECIPIENT_CMV_SEROLOGY": {
        0: "Negative",
        1: "Positive",
        -1: "Not Done / Not Reported / Missing",
    },
    "DIABETES": {0: "No", 1: "Yes"},
    "VENOUS_THROMBOEMBOLISM": {0: "No", 1: "Yes"},
    "PULM_EMBOLISM": {0: "No", 1: "Yes"},
    "SUPPLEMENTAL_OXYGEN": {0: "No", 1: "Yes"},
    "VENTILATOR": {0: "No", 1: "Yes"},
    "ECMO": {0: "No", 1: "Yes"},
    "PGD_Grade_3": {0: "No", 1: "Yes"},
    "DSA_present": {0: "No", 1: "Yes"},
    "AMR_present": {0: "No", 1: "Yes"},
    "SMOKING_HISTORY_RECIPIENT": {0: "No", 1: "Yes"},
    "Race_mismatch": {0: "No", 1: "Yes"},
    "Blood_type_mismatch": {0: "No", 1: "Yes"},
    "RACE": {0: "White", 1: "Black", 2: "Others", -1: "Missing"},
    "BLOOD_TYPE_RECIPIENT": {0: "O", 1: "A", 2: "B", 3: "AB", -1: "Missing"},
    "NATIVE_LUNG_DISEASE_Coded": {
        1: "Cystic Fibrosis",
        2: "Idiopathic pulmonary fibrosis",
        3: "COPD",
        4: "Sarcoidosis",
        5: "Idiopathic pulmonary arterial hypertension",
        6: "Non-CF bronchiectasis",
        7: "Other ILDs",
        8: "Other, specific",
    },
}

FEATURE_LABEL_REVERSE_MAP = {label: name for name, label in FEATURE_LABEL_MAP.items()}
