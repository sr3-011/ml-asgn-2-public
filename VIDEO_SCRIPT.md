# Team 13 — Assignment 2 Video Script
## BITS F464 Machine Learning | Semester 2 2025-26

---

## Team Details (show at start of video)
- **Team Number:** 13
- **Course:** BITS F464 Machine Learning
- **Campus:** BITS Pilani Hyderabad Campus
- **Semester:** Second Semester 2025-26
- **Members:**
  - Shriniketh — Data Architect
  - Sanvi — ML Engineer
  - Dheeraj — Full Stack Developer
  - Shambhavi — Analyst

---

## Before You Start Recording
- Make sure the dashboard is running:
  ```
  streamlit run Team13_Assignment2_dashboard.py
  ```
- Open browser at http://localhost:8501
- Set browser zoom to 100%
- Use a screen recorder (OBS, Loom, or QuickTime)
- **Recommended video length:** 8–12 minutes
- Record in at least 1080p

---

## Section 1 — Introduction (30–45 seconds)
**SHOW:** Title slide or just introduce verbally

**SAY:**
> "Hi, we are Team 13 from BITS Pilani Hyderabad Campus for the course BITS F464 Machine Learning. This video is a walkthrough of our Assignment 2 — Automated Machine Learning Pipeline for Clinical Prediction under Temporal Shift in EHR Data."

---

## Section 2 — Project Overview Page (1–2 minutes)
**SHOW:** Navigate to the Project Overview page in the sidebar

**WHAT TO CLICK:**
- Click **"Project Overview"** in the left sidebar

**WHAT TO EXPLAIN:**
- "The dataset is a synthetic EHR dataset with 17 tables covering patient demographics, clinical observations, diagnoses, medications, procedures and more."
- Point to the **4 metric cards** at the top — explain patient counts, feature count, D1 vs D2 sizes.
- "We split the data temporally — patients with encounters before 2020 form Dataset 1 (Historical) and patients with encounters from 2020 onwards form Dataset 2 (Current). This simulates real-world temporal shift in healthcare data."
- Point to the **Pipeline Architecture** section and briefly explain the two datasets.
- Point to the **Team** section and name each member and their role.

---

## Section 3 — EDA Page (2–3 minutes)
**SHOW:** Navigate to the Exploratory Data Analysis page

**WHAT TO CLICK AND EXPLAIN** (go through each dropdown option):

**1. Class Distribution**
- Click **"Class Distribution"** in the dropdown.
- EXPLAIN: "This shows the severe class imbalance — about 95% of patients have no diagnosed condition and only 5% do. This was a key challenge we had to address using `class_weight=balanced` for Decision Tree and SVM, and SMOTE oversampling for the Neural Network."

**2. Demographics**
- Click **"Demographics"** in the dropdown.
- EXPLAIN: "Here we can see demographic distributions split by label. Age, gender, race and income are shown for both historical and current datasets."
- Select a few demographics from the inner dropdown to show.

**3. Clinical Features**
- Click **"Clinical Features"** in the dropdown.
- EXPLAIN: "These violin plots show clinical measurements like BMI, blood pressure, heart rate and cholesterol split by whether the patient has a condition or not. You can see slight differences between the two groups."
- Select 2–3 features from the inner dropdown.

**4. Healthcare Utilization**
- Click **"Healthcare Utilization"** in the dropdown.
- EXPLAIN: "Patients with conditions tend to have more encounters, medications and procedures — which makes clinical sense."

**5. Correlation Heatmap**
- Click **"Correlation Heatmap"** in the dropdown.
- EXPLAIN: "All correlations with the label are weak — under 0.1. This is expected given the class imbalance and confirms that no single feature is strongly predictive on its own."

**6. Data Drift Analysis**
- Click **"Data Drift Analysis"** in the dropdown.
- EXPLAIN: "This compares the distribution of top features between Dataset 1 and Dataset 2. The distributions look similar post-scaling, which means the temporal shift is subtle rather than dramatic."

**7. Missing Values**
- Click **"Missing Values"** in the dropdown.
- EXPLAIN: "132 columns were dropped due to over 50% missingness — mostly rare lab tests and IgE allergy panels. The 104 surviving features had good coverage across patients."

---

## Section 4 — Model Performance Page (2–3 minutes)
**SHOW:** Navigate to the Model Performance page

**WHAT TO CLICK AND EXPLAIN:**

**1. Metric selector overview**
- Show the metric selector dropdown at the top.
- EXPLAIN: "We trained three models — Decision Tree, SVM, and Neural Network — on Dataset 1's training set. We then evaluated each model on both the Dataset 1 test set and Dataset 2 test set to measure how well they generalize across time."

**2. F1 Score**
- Select **"f1"** from the metric dropdown.
- EXPLAIN: "F1 score is our primary metric because accuracy is misleading with a 95/5 class imbalance — a model predicting all zeros gets 95% accuracy but is useless. The MLP achieves the best F1 on Dataset 2."

**3. ROC-AUC**
- Select **"roc_auc"** from the metric dropdown.
- EXPLAIN: "ROC-AUC tells us how well the model separates the two classes. The MLP achieves 0.90 on Dataset 2, which is strong."

**4. Accuracy**
- Select **"accuracy"** from the metric dropdown.
- EXPLAIN: "Notice how accuracy looks great for all models — this is exactly why accuracy alone is misleading for imbalanced datasets."

**5. Full Metrics Table**
- Scroll down to the **Full Metrics Table**.
- EXPLAIN: "This table shows all 5 metrics for all 6 combinations of model and dataset. The highlighted cells show the best F1 and ROC-AUC scores."

**6. ROC Curves**
- Scroll down to **ROC Curves**.
- EXPLAIN: "ROC curves for all three models on both datasets. The MLP curve is closest to the top-left corner, indicating the best performance."

**7. Confusion Matrices**
- Scroll down to **Confusion Matrices**.
- EXPLAIN: "These 6 confusion matrices show true positives, false positives, true negatives and false negatives for each model on each dataset."

**8. Analysis section**
- Scroll down to the **Analysis** section.
- EXPLAIN: "The Decision Tree has high bias — it underfits slightly. The SVM performs poorly on D1 but recovers on D2. The MLP with SMOTE gives the best overall generalization."

---

## Section 5 — Continual Learning Page (1–2 minutes)
**SHOW:** Navigate to the Continual Learning page

**WHAT TO CLICK AND EXPLAIN:**

**1. Before/after metric cards**
- Show the before/after metric cards.
- EXPLAIN: "We implemented continual learning by fine-tuning the D1-trained MLP on Dataset 2 training data using `partial_fit` over 50 epochs."

**2. Delta values**
- Point to the delta values on the metric cards.
- EXPLAIN: "Interestingly, continual learning actually hurt performance — F1 dropped significantly. This is a well-known phenomenon called catastrophic forgetting, where fine-tuning on new data overwrites previously learned weights."

**3. Comparison bar chart**
- Show the **comparison bar chart**.
- EXPLAIN: "This chart clearly shows the drop across all metrics after continual learning."

**4. Analysis section**
- Scroll down to the analysis section.
- EXPLAIN: "The key insight here is that our MLP trained on historical data already generalized well to current data — achieving strong F1 and ROC-AUC. The aggressive `partial_fit` approach without regularization caused forgetting. More sophisticated techniques like Elastic Weight Consolidation would be needed in a real scenario."

---

## Section 6 — Feature Importance Page (1 minute)
**SHOW:** Navigate to the Feature Importance page

**WHAT TO CLICK AND EXPLAIN:**

**1. Top 20 feature importance chart**
- Show the top 20 feature importance bar chart.
- EXPLAIN: "The Decision Tree relies most heavily on observation-derived features — aggregated vitals and lab results like BMI, blood pressure and cholesterol. These make clinical sense as predictors of diagnosed conditions."

**2. Feature category metrics**
- Point to the 4 feature category metric cards.
- EXPLAIN: "We have 4 categories of features — demographic, encounter aggregates, clinical observations, and utilization features. The observation features dominate."

**3. Search box demo**
- Type something in the search box to demonstrate it.
- EXPLAIN: "This searchable list lets you explore all 104 features used by the models."

**4. Interpretation section**
- Scroll to the interpretation section.
- EXPLAIN: "Different models use features differently — the Decision Tree is interpretable and uses a small subset, while the SVM and MLP use all features through more complex transformations."

---

## Section 7 — Closing Summary (30–45 seconds)
**SHOW:** Go back to Project Overview or stay on the last page

**SAY:**
> "To summarise — we built an end-to-end ML pipeline on a synthetic EHR dataset. We merged 15 tables, engineered 104 features, and trained three classifiers to predict clinical conditions. The MLP with SMOTE oversampling achieved the best performance with strong F1 and ROC-AUC scores on both historical and current data. We demonstrated temporal shift analysis and found that our D1-trained model generalizes well to D2 data. Our continual learning experiment revealed catastrophic forgetting, highlighting the need for more advanced techniques in production healthcare systems. Thank you."

---

## Recording Tips
- Speak clearly and at a moderate pace
- Pause briefly before clicking into each new section
- Make sure plot labels are visible on screen before explaining them
- If something takes time to load, just keep talking
- Aim for 8–12 minutes total — do not rush
- Export as MP4 named: `Team13_Assignment2_video.mp4`
