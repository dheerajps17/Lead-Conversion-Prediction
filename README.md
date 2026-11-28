# Lead Conversion Prediction using Machine Learning

A binary classification project that predicts whether a lead will convert into a customer, so sales teams can prioritise high-potential leads and focus their follow-ups.

## Problem Statement

A company collects leads from several campaigns and sources (Instagram, Facebook Ads, file uploads, walk-ins). Each lead moves through the sales pipeline and may or may not convert. The goal is to build a model that estimates the likelihood of conversion from historical lead data.

## Dataset

- **File:** `Unified_Leads.csv` (27,375 leads, 15 columns)
- **Key columns:** `creation_date`, `first_call_attempt`, `lead_status`, `lead_stage`, `lead_tag`, `assigned_to`, `campaign_name`, `source_type`, `source_name`
- Names, phone numbers and emails are anonymised.

## Approach

### 1. Data preprocessing
- Dropped the empty `references` column and merged duplicate salesperson names.
- Parsed date columns; created a `called` flag from missing call dates.
- **Target:** `CONVERTED` = 1, `LOST` = 0. Leads still at *Prospect* or *Follow Up Required* (14,832) have no outcome, so they are excluded from training and scored afterwards.
- Training set: 12,543 resolved leads, 6.7% converted (imbalanced).
- **Leakage removed:** `lead_status` and `lead_tag` are filled in after the outcome is known, so they are not used as features.
- Features engineered: creation hour, weekday, weekend flag, hours to first call, first-call hour. Rare categories grouped into "Other"; categorical variables one-hot encoded.

### 2. Exploratory data analysis
- The **Business Accounting** campaign converts at about 32.8%; all other campaigns sit between 1.5% and 3.2%.
- Leads assigned to **Anshad** convert at about 52%, against 4% or less for everyone else (likely a closing or manager account, so this is not necessarily skill).
- Instagram (7.5%) and Facebook Ads (6.0%) outperform File Upload (3.3%) and Move Lead (1.6%).
- Older leads show higher conversion partly because newer leads are still in progress.

### 3. Models trained
Logistic Regression, Decision Tree and Random Forest (scikit-learn), with class weights for the imbalance.

### 4. Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.84 | 0.13| 0.72 | 0.22 | 0.87 |
| Decision Tree | 0.88 | 0.17 | 0.66 | 0.27 | 0.83 |
| Random Forest | 0.96 | 0.50 | 0.48 | 0.49 | 0.85 |

### 5. Prediction system
- All 14,832 open leads are scored and ranked High / Medium / Low priority (`open_leads_prioritised.csv`).
- `predict_new_lead()` returns a conversion probability and a label for a single new lead.

## Repository structure

```
├── lead_conversion.py            # full pipeline: preprocessing, EDA, training, evaluation, prediction
├── Unified_Leads.csv             # dataset
├── lead_conversion_model.joblib          
├── eda_conversion_rates.png
├── model_evaluation.png
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python lead_conversion.py
```

Update the input and output paths at the top of the script to match your folders.

### Predict a new lead

```python
predict_new_lead(
    source_type="INSTAGRAM",
    source_name="INSTAGRAM",
    campaign_name="Business Accounting",
    assigned_to="Megha S",
    created_at="2026-02-05 11:00",
    first_call_at="2026-02-05 12:00",   # optional
)
# {'conversion_probability': ..., 'prediction': 'WILL CONVERT' / "WON'T CONVERT"}
```

## Limitations

- The model relies heavily on who handled the lead and which campaign it came from, so it may weaken if staffing or campaigns change.
- Only 845 converted leads are available, so rare segments have limited evidence.
- The 0.80 threshold was tuned on the same test set, so its metrics are slightly optimistic.
- Useful extra data: call duration, number of follow-ups, and actual conversion dates.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib
