# ML Assignment 2 — EHR Patient Classification

**Team:** [Team Name / Number]

Binary classification on a synthetic EHR dataset to predict whether a patient has any disorder/finding condition. Two temporally-split datasets are used to evaluate model performance and data drift.

---

## Repository Structure

```
ml_asgn_2/
├── pipeline.py                        # Data ingestion, feature engineering, train/test split
├── eda.py                             # EDA plots and drift analysis
├── TeamXX_Assignment2_dashboard.py    # Streamlit dashboard (5 pages)
├── data/
│   ├── *.csv                # Raw EHR tables (17 files — gitignored, download separately)
│   ├── processed/           # Pickled train/test splits + scalers
│   │   ├── X_train_d1.pkl, X_test_d1.pkl, y_train_d1.pkl, y_test_d1.pkl
│   │   ├── X_train_d2.pkl, X_test_d2.pkl, y_train_d2.pkl, y_test_d2.pkl
│   │   ├── scaler_d1.pkl, scaler_d2.pkl
│   │   └── feature_names.pkl
│   └── eda/                 # 28 PNG plots + drift stats CSV
└── models/                  # Trained model .pkl files + metrics CSVs + plots
    ├── metrics_baseline.csv
    ├── metrics_continual.csv
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance_dt.png
    └── continual_learning_comparison.png
```

---

## Setup

**Python 3.11 is required.** The pickle files in `data/processed/` were generated with Python 3.11 + NumPy 2.x — using a different Python version will cause unpickling errors.

### 1. Clone the repo

```bash
git clone https://github.com/<your-org>/mlasgn2public.git
cd mlasgn2public
```

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt)**
```bat
py -3.11 -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run this once as Administrator:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Your terminal prompt will show `(venv)` when the environment is active. Run `deactivate` to exit it.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the `data/` directory and add the raw CSV files

The app reads CSV files from a `data/` folder in the repo root. This folder is **not** included in the repository — you need to create it and populate it manually.

**Step 1 — Create the folder**

macOS / Linux:
```bash
mkdir data
```

Windows (Command Prompt or PowerShell):
```bat
mkdir data
```

**Step 2 — Copy the CSV files into `data/`**

Place all raw EHR CSV files directly inside `data/` (no subdirectories). The dashboard expects these 15 files:

```
data/
├── patients.csv
├── encounters.csv
├── observations.csv
├── conditions.csv
├── medications.csv
├── procedures.csv
├── immunizations.csv
├── allergies.csv
├── careplans.csv
├── imaging_studies.csv
├── devices.csv
├── supplies.csv
├── payer_transitions.csv
├── claims.csv
└── claims_transactions.csv
```

Your repo root should then look like:

```
mlasgn2public/
├── data/
│   ├── patients.csv
│   ├── encounters.csv
│   └── ... (15 CSVs total)
├── Team13_Assignment2_dashboard.py
└── requirements.txt
```

> The CSV files are distributed via the shared Google Drive link provided by the course. Do not commit them to the repository — `data/` is already listed in `.gitignore`.

---

## Running the Dashboard

Make sure you are in the repo root directory, then run:

```bash
source venv/bin/activate && streamlit run Team13_Assignment2_dashboard.py
```

On Windows:
```bat
venv\Scripts\activate.bat && streamlit run Team13_Assignment2_dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The dashboard has 5 pages (navigate via the left sidebar):

| Page | Contents |
|------|----------|
| Project Overview | Pipeline architecture, team roles, dataset summary |
| Exploratory Data Analysis | Class distribution, demographics, clinical features, drift, missing values |
| Model Performance | Per-metric bar charts, full metrics table, ROC curves, confusion matrices |
| Continual Learning | MLP before/after fine-tuning, catastrophic forgetting analysis |
| Feature Importance | Decision Tree top-20 features, feature category breakdown, searchable list |

> **Important:** Always run `streamlit run` from the repo root (the folder containing `data/`). Running it from a different directory will cause a `FileNotFoundError` because the app looks for `data/` relative to the working directory.


---

## Datasets

| Split | Shape | Label=0 | Label=1 |
|-------|-------|---------|---------|
| D1 Train | (1721, 104) | 1639 (95.2%) | 82 (4.8%) |
| D1 Test  | (431, 104)  | 410 (95.1%)  | 21 (4.9%) |
| D2 Train | (1580, 104) | 1502 (95.1%) | 78 (4.9%) |
| D2 Test  | (395, 104)  | 375 (94.9%)  | 20 (5.1%) |

**Temporal split:** Dataset 1 = patients whose earliest encounter is before 2020-01-01. Dataset 2 = patients with any encounter on or after 2020-01-01.

---

## Team Responsibilities

### Data Architect — Shriniketh 
Deliverables in `data/processed/` and `data/eda/`. Done.

### ML Engineer - Sanvi
- Load splits from `data/processed/`
- Train Decision Tree, SVM, Neural Network
- Evaluate on **both** D1 test and D2 test sets
- Save trained models as `.pkl` + a metrics CSV

### Full Stack Developer - Dheeraj
- Build `TeamXX_Assignment2_dashboard.py` (Streamlit)
- Inputs: model outputs + EDA plots from `data/eda/`

### Analyst - Shambhavi
- EDA plots: `data/eda/`
- Drift stats: `data/eda/eda_drift_stats.csv` (generated by `eda.py`)
- Produce written analysis and `video.mp4`

---

## Critical Notes for All Teammates

> **Class imbalance:** ~95% negative, ~5% positive. Use `class_weight='balanced'` for Decision Tree and SVM. For Neural Network, use weighted loss or SMOTE.

> **Do not re-scale the data.** All splits are already StandardScaled (fit on train, applied to test). Loading from `data/processed/` gives you ready-to-use arrays.

> **Feature names:** `feature_names.pkl` contains the ordered list of all 104 features.

> **Metrics:** Report F1-score and ROC-AUC — accuracy is misleading given the imbalance.

> **Reproducibility:** Use `random_state=42` everywhere.
