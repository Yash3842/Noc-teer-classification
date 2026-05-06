# NOC TEER Classification Pipeline

> **Automated job title → occupational classification for 4.7M+ loan applicants using NLP, fuzzy matching, and calibrated Random Forest models.**  
> Built during an 8-month ML Engineering co-op at goeasy Ltd.

---

## What is TEER?

Canada's National Occupational Classification (NOC) system assigns every job two values:

| Code | Name | Range | Meaning |
|------|------|-------|---------|
| **Hierarchical TEER** | Education level | 0–5 | 0 = Management, 5 = No formal education |
| **Field TEER** | Occupational category | 0–9 | 2 = Science/Tech, 6 = Sales/Service, etc. |

Predicting these from raw, messy, free-text job titles at scale is the core problem this pipeline solves.

---

## Pipeline Overview

```
Raw job titles (messy, bilingual, noisy)
        │
        ▼
┌─────────────────────────┐
│  Step 1: Data Cleaning  │  Hybrid French detection (regex + langdetect)
│                         │  Gibberish flagging (placeholders, non-job titles)
│                         │  Text normalization (lowercase, strip special chars)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 2: NOC Matching   │  16 override rules for common titles
│                         │  Fuzzy match vs 27K official NOC titles (rapidfuzz)
│                         │  Penalty scoring · 4-thread parallel processing
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 3: ML Training    │  TF-IDF features: job title + employer + city
│                         │  RandomForest (500 trees) + isotonic calibration
│                         │  Trained on high-confidence matches (score ≥ 85)
└────────────┬────────────┘
             │
             ▼
        TEER Predictions
  (Hierarchical + Field, with confidence)
```

---

## Results

| Dataset | Hierarchical TEER | Field TEER | Avg Confidence |
|---------|-------------------|------------|----------------|
| Training set (score ≥ 85) | **95.08%** | **95.61%** | ~92% |
| All predictions (score ≥ 75) | **77.56%** | **78.97%** | ~84% |

---

## Tech Stack

- **PySpark** — distributed data processing on Databricks
- **rapidfuzz** — fast fuzzy string matching against 27K NOC titles
- **scikit-learn** — TF-IDF vectorization, RandomForest, CalibratedClassifierCV
- **MLflow + Unity Catalog** — model versioning and registry
- **langdetect** — bilingual (EN/FR) title filtering

---

## Key Design Decisions

**Why fuzzy matching before ML?**  
Fuzzy matching gives interpretable, rule-based labels that become high-quality training data. The ML model then generalizes to novel titles the fuzzy matcher struggles with.

**Why calibrate the Random Forest?**  
Raw RF confidence scores are poorly calibrated (overconfident). Isotonic regression calibration makes the confidence scores actually mean what they say — critical when routing edge cases to human review.

**Why deduplicate before fuzzy matching?**  
With 4.7M records but far fewer unique titles, deduplicating first and broadcasting results cuts fuzzy matching time by orders of magnitude.

**French title handling**  
A hybrid approach (Unicode character detection + langdetect + allowlist of unambiguous French words) prevents both false positives (legitimate English titles flagged as French) and false negatives.

---

## Feature Engineering

```python
FEATURE_COLS = [
    "cleaned_emp_position",    # TF-IDF (15K features, 1–3 ngrams)
    "ApplicantEmployersName",  # TF-IDF (5K features, 1–2 ngrams)
    "ApplicantCity",           # TF-IDF (2K features, 1–2 ngrams)
    "yearly_salary",           # Numeric (StandardScaler)
    "score",                   # Fuzzy match confidence
    "title_word_count",        # Derived
    "title_char_length"        # Derived
]
```

---

## Repository Structure

```
noc-teer-classification/
├── noc_pipeline.py          # Full pipeline: cleaning → matching → training → prediction
├── README.md
└── sample_noc_titles.csv    # Sample of public NOC title reference data (no customer data)
```

> ⚠️ **Note:** No customer data is included in this repository. The pipeline was trained on proprietary applicant data. The code is shared for portfolio/reference purposes.

---

## Model Architecture

```
ColumnTransformer
├── TfidfVectorizer("cleaned_emp_position", max_features=15000, ngram_range=(1,3))
├── TfidfVectorizer("ApplicantEmployersName", max_features=5000, ngram_range=(1,2))
├── TfidfVectorizer("ApplicantCity", max_features=2000, ngram_range=(1,2))
└── StandardScaler(["yearly_salary", "score", "title_word_count", "title_char_length"])
        │
        ▼
RandomForestClassifier(n_estimators=500, max_depth=30, class_weight="balanced")
        │
        ▼
CalibratedClassifierCV(method="isotonic")  ← key for reliable confidence scores
```

Two separate models trained: one for **Hierarchical TEER** (education), one for **Field TEER** (occupational category).

---

## Context

Built at **goeasy Ltd.** (8-month ML Engineering co-op, Fall 2024) as part of the Risk team's automated loan decisioning pipeline. The model processes applicant job titles to infer education level and occupational category — signals used downstream in credit risk assessment.

---

*For questions or collaborations → [linkedin.com/in/yash-patel-35449a226](https://www.linkedin.com/in/yash-patel-35449a226/)*
