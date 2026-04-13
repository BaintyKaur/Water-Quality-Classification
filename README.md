# 💧 Water Quality Classification for Safe Drinking Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **Machine Learning-based Water Potability Prediction System**  
> Predicts whether a water sample is safe to drink using advanced classification models

---

## 🔗 Live Demo

**[🌐 Open the Streamlit App →](https://your-app-name.streamlit.app)**

*(Deploy on Streamlit Community Cloud after completing all stages)*

To run locally: `streamlit run app.py`

---

## 👥 Team Members

| Name | GitHub | Responsibilities |
|------|--------|------------------|
| Bainty Kaur | [@BaintyKaur](https://github.com/BaintyKaur) | Data Collection, Preprocessing, EDA, Feature Engineering |
| Asna Abbas | [@asnaabbas763-cyber](https://github.com/asnaabbas763-cyber) | Model Building, Evaluation, Deployment, Explainability |

---

## 📌 Problem Statement

Access to safe drinking water is a critical global health issue. Over **2 billion people** consume contaminated water worldwide, leading to diseases like cholera, typhoid, and dysentery.

Traditional water quality testing is expensive, slow, and requires laboratory equipment. This project builds a **machine learning-based binary classifier** that predicts whether a water sample is **potable (safe to drink)** or **non-potable** based on physicochemical properties — enabling fast, scalable water safety assessment.

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| **Name** | Water Potability Dataset |
| **Source** | [Kaggle — Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability) |
| **Size** | 3,276 samples × 10 features |
| **Target** | `Potability` — 1 (Safe), 0 (Not Safe) |
| **Class Distribution** | ~61% Non-Potable, ~39% Potable (imbalanced) |

### Features

| Feature | Description | WHO Safe Limit |
|---------|-------------|----------------|
| `ph` | pH of water | 6.5 – 8.5 |
| `Hardness` | Calcium/Magnesium content (mg/L) | < 300 |
| `Solids` | Total dissolved solids (ppm) | < 500 |
| `Chloramines` | Disinfectant concentration (ppm) | < 4 |
| `Sulfate` | Sulfate ions (mg/L) | < 250 |
| `Conductivity` | Electrical conductivity (μS/cm) | < 400 |
| `Organic_carbon` | Organic matter (ppm) | < 2 |
| `Trihalomethanes` | THM disinfection byproducts (μg/L) | < 80 |
| `Turbidity` | Water clarity (NTU) | < 5 |

---

## 🔬 Project Workflow — 8 Stages

This project follows a structured 8-stage machine learning pipeline:

### Stage 1 — Problem Definition & EDA Preparation
- **File:** [01_problem_definition.ipynb](01_problem_definition.ipynb)
- Defined binary classification problem: potable (safe) vs. non-potable (unsafe)
- Reviewed water quality standards and health implications
- Set evaluation metrics: **F1-score** (primary, due to class imbalance), Accuracy, Precision, Recall, ROC-AUC

### Stage 2 — Data Collection & Exploration
- **File:** [02_data_collection.ipynb](02_data_collection.ipynb)
- Source: Water Potability Dataset from Kaggle
- Loaded raw data: 3,276 samples × 10 features
- Identified missing values: `ph` (491), `Sulfate` (781), `Trihalomethanes` (162)
- Class distribution: ~61% Non-Potable, ~39% Potable (class imbalance detected)

### Stage 3 — Data Preprocessing & Cleaning
- **File:** [03_preprocessing.ipynb](03_preprocessing.ipynb)
- **Missing values:** Class-wise median imputation (preserves class statistics better than global median)
- **Outliers:** IQR-based Winsorization (capping extreme values) to preserve data magnitude while reducing influence
- **Duplicates:** Verified zero duplicate rows
- **Train-Test Split:** 80-20 split with stratification to maintain class balance
- **Output:** `data/processed/water_potability_cleaned.csv`

### Stage 4 — Exploratory Data Analysis
- **File:** [04_eda.ipynb](04_eda.ipynb)
- Distribution plots with KDE curves (overlaid by Potability class)
- Correlation heatmap (Pearson correlation, visualized with clustermap)
- Violin plots showing feature distributions per class
- Statistical tests: Mann-Whitney U test for all features (identify significant predictors)
- **Finding:** Features show significant overlap between classes → non-linear models likely needed

### Stage 5 — Feature Engineering & Selection
- **File:** [05_feature_engineering.ipynb](05_feature_engineering.ipynb)
- **Created 14 new engineered features:**
  - WHO safety binary flags (9 features) — flag if each parameter meets WHO safe limits
  - `total_who_compliant` — count of WHO-compliant parameters
  - Ratio features: `hardness_to_conductivity`, `chloramines_to_thm`, `organic_to_turbidity`
  - `ph_deviation` — absolute distance from neutral pH 7.0
  - Log transforms: `log_solids`, `log_conductivity`
- **Feature selection:** Random Forest Gini importance (threshold: 0.01) + Mutual Information ranking
- **Standardization:** StandardScaler applied to numeric features
- **Output:** `data/processed/water_potability_engineered.csv`

### Stage 6 — Model Building & Hyperparameter Tuning
- **File:** [06_model_building.ipynb](06_model_building.ipynb)
- Trained **3 distinct models** with GridSearchCV (5-fold cross-validation):

| Model | Hyperparameters Tuned |
|-------|----------------------|
| **Random Forest** | n_estimators (100, 200), max_depth (5-20), min_samples_split (2-10), max_features ("sqrt", "log2") |
| **SVM** | C (0.1-100), kernel (rbf, linear), gamma (scale, auto) |
| **XGBoost** | n_estimators (100-500), max_depth (3-10), learning_rate (0.01-0.3), subsample (0.7-1.0) |

- **Class imbalance handled:** SMOTE applied to training data to oversample minority class
- **Scaler saved:** StandardScaler object saved to `models/scaler.pkl` for consistent test-time scaling
- **Models saved:** Each trained model pickled to `models/{model_name}.pkl`

### Stage 7 — Model Evaluation & Comparison
- **File:** [07_model_evaluation.ipynb](07_model_evaluation.ipynb)
- Evaluated all 3 models on held-out 20% test set:
  - Classification reports (Precision, Recall, F1-score per class)
  - Confusion matrices
  - ROC-AUC curves with AUC scores
  - Precision-Recall curves
  - Cross-validation score distributions
- **Best model selection:** Automatically selected based on ROC-AUC
- **Output:** Best model name saved to `models/best_model_name.pkl`

### Stage 8 — Model Interpretation & Explainability
- **File:** [08_explainability.ipynb](08_explainability.ipynb)
- **SHAP (SHapley Additive exPlanations)** analysis:
  - Global explanation: Summary plots showing feature importance across entire test set
  - Local explanation: Waterfall plots for individual predictions (potable vs. non-potable samples)
  - Dependence plots: Feature interaction insights for top 3 predictive features
  - **Actionable insights:** Derive specific environmental monitoring recommendations
- **Visualizations saved** to `data/processed/` for use in Streamlit app

---

## � Dataset Overview

| Property | Detail |
|----------|--------|
| **Name** | Water Potability Dataset |
| **Source** | [Kaggle — Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability) |
| **Size** | 3,276 samples × 10 original features |
| **Target** | `Potability` — 1 (Potable/Safe), 0 (Non-Potable/Unsafe) |
| **Class Distribution** | ~61% Non-Potable, ~39% Potable (imbalanced) |
| **Train-Test Split** | 80% training (with SMOTE applied), 20% test |

### Original Features (9)

| Feature | Unit | Description | WHO Safe Limit |
|---------|------|-------------|---|
| `ph` | — | Acidity/alkalinity | 6.5 – 8.5 |
| `Hardness` | mg/L | Calcium/Magnesium content | < 300 |
| `Solids` | ppm | Total dissolved solids | < 500 |
| `Chloramines` | ppm | Disinfectant concentration | < 4 |
| `Sulfate` | mg/L | Sulfate ions | < 250 |
| `Conductivity` | μS/cm | Electrical conductivity | < 400 |
| `Organic_carbon` | ppm | Organic matter | < 2 |
| `Trihalomethanes` | μg/L | Disinfection byproducts | < 80 |
| `Turbidity` | NTU | Water clarity | < 5 |

---

## 📸 Application Screenshots

The **Streamlit web app** provides an interactive interface with:
- **Real-time Predictions:** Input water sample parameters via sliders
- **Multi-model Support:** Compare predictions across Random Forest, SVM, and XGBoost
- **WHO Compliance Checker:** Visualize which parameters meet WHO safe limits  
- **Prediction Confidence:** Display probabilities for both potable/non-potable classes
- **SHAP Explainability:** Interactive plots explaining individual predictions
- **Evaluation Metrics:** Confusion matrices, ROC curves, and performance dashboards

*(Screenshots will be added after deployment)*

---

## 🗂️ Repository Structure

## 🗂️ Repository Structure

```
Water-Quality-Classification/
├── 01_problem_definition.ipynb          ← Stage 1: Problem definition
├── 02_data_collection.ipynb             ← Stage 2: Data loading & exploration
├── 03_preprocessing.ipynb               ← Stage 3: Cleaning & preprocessing
├── 04_eda.ipynb                         ← Stage 4: Exploratory data analysis
├── 05_feature_engineering.ipynb         ← Stage 5: Feature creation & selection
├── 06_model_building.ipynb              ← Stage 6: Model training & tuning
├── 07_model_evaluation.ipynb            ← Stage 7: Model evaluation & comparison
├── 08_explainability.ipynb              ← Stage 8: SHAP analysis
├── app.py                               ← Streamlit web application
├── requirements.txt                     ← Python dependencies
├── README.md                            ← This file
├── GIT_WORKFLOW.md                      ← Team collaboration guide
├── .gitignore                           ← Git ignore rules
│
├── data/
│   ├── raw/
│   │   └── water_potability.csv         ← Original dataset from Kaggle
│   └── processed/
│       ├── water_potability_cleaned.csv         ← After preprocessing (Stage 3)
│       ├── water_potability_engineered.csv     ← After feature engineering (Stage 5)
│       ├── X_train.csv, X_train_scaled.csv     ← Training features (scaled)
│       ├── X_test.csv, X_test_scaled.csv       ← Test features (scaled)
│       ├── y_train.csv, y_test.csv             ← Training/test labels
│       └── water_potability_raw_copy.csv       ← Backup copy
│
├── models/
│   ├── random_forest.pkl                ← Trained Random Forest model
│   ├── svm.pkl                          ← Trained SVM model
│   ├── xgboost.pkl                      ← Trained XGBoost model
│   ├── best_model.pkl                   ← Best performing model
│   ├── best_model_name.pkl              ← Name of best model
│   ├── scaler.pkl                       ← StandardScaler (fit on train data)
│   └── selected_features.pkl            ← Selected feature names
│
├── .venv/                               ← Python virtual environment (gitignored)
└── .git/                                ← Git repository metadata
```

---

## 🚀 Quick Start

### Option 1: Run Streamlit App (Fastest)

```bash
# 1. Navigate to project directory
cd Water-Quality-Classification

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (if not already present)
# Visit: https://www.kaggle.com/datasets/adityakadiwal/water-potability
# Save to: data/raw/water_potability.csv

# 5. Run Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

### Option 2: Run the Notebooks (Full Workflow)

```bash
# 1–3. Same setup as above, then:

# 4. Start Jupyter notebook server
jupyter notebook

# 5. Open and run notebooks in order (01 → 08)
# Each notebook builds on the previous outputs
```

### Option 3: Deploy to Streamlit Cloud (Production)

1. **Push code to GitHub** (if not already done)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click **"New app"** → Select your repository, branch, and `app.py`
4. Deploy! Your app will have a URL: `https://your-username-projectname.streamlit.app`

---

## 📋 Prerequisites & Dependencies

- **Python:** 3.10 or higher
- **Git:** For version control
- **Kaggle Account:** To download the dataset

**Key Python Packages:**
- `pandas`, `numpy`, `scipy` — Data manipulation & statistics
- `scikit-learn`, `xgboost`, `lightgbm` — Machine learning models
- `imbalanced-learn` — SMOTE for handling class imbalance
- `shap` — Model explainability
- `matplotlib`, `seaborn`, `plotly` — Data visualization
- `streamlit` — Web app framework
- `joblib` — Model serialization

All dependencies listed in [requirements.txt](requirements.txt)

---

## 🤝 Git Collaboration & Team Workflow

See [GIT_WORKFLOW.md](GIT_WORKFLOW.md) for detailed instructions on:
- Initial repository setup
- Creating feature branches
- Committing with meaningful messages
- Opening and reviewing Pull Requests
- Using GitHub Issues for task tracking

**Quick Summary of Branch Strategy:**

```bash
# Each team member creates their own feature branch
git checkout -b feature/stage-N-description

# Example:
git checkout -b feature/stage-4-eda-analysis
git checkout -b feature/stage-6-model-training

# After finishing work, create a PR for review before merging to main
```

**Commit Message Convention:**
```bash
git commit -m "feat(stage-N): Brief description of changes

- Detailed point 1
- Detailed point 2
- Files saved to: path/to/output"
```

---

## 📊 Key Results & Findings

After running all stages:

1. **Best Model Performance:**
   - Compare evaluation metrics from `07_model_evaluation.ipynb`
   - ROC-AUC, F1-score, and Precision-Recall curves

2. **Feature Importance:**
   - Top predictive features from Random Forest feature importance
   - SHAP summary plots from `08_explainability.ipynb`

3. **Class Imbalance Impact:**
   - SMOTE improved recall for minority (potable) class
   - F1-score more reliable metric than accuracy

4. **WHO Compliance Insights:**
   - Engineered WHO-compliance flags are strong predictors
   - pH and sulfate levels are critical quality indicators

---

## 🔍 Model Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to interpret model predictions:

- **Global Explanations:** Which features matter most across all predictions
- **Local Explanations:** Why a specific water sample was predicted potable/non-potable
- **Dependence Plots:** How feature values affect predictions
- **Waterfall Plots:** Breakdown of prediction components

All SHAP visualizations are generated in [08_explainability.ipynb](08_explainability.ipynb) and displayed in the Streamlit app.

---

## 🐛 Troubleshooting

### Issue: Models not loading in Streamlit app
**Solution:** Ensure all `.pkl` files exist in the `models/` directory. Run notebooks 01–07 first.

### Issue: Missing data files
**Solution:** Download `water_potability.csv` from [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability) and place in `data/raw/`

### Issue: SMOTE or Scaler errors
**Solution:** Ensure `imbalanced-learn` is installed: `pip install imbalanced-learn`

### Issue: Streamlit port already in use
**Solution:** Use a different port: `streamlit run app.py --server.port 8502`

---

## 📚 Technologies & Libraries

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.10+ |
| **Data Science** | pandas, numpy, scipy, scikit-learn |
| **ML Models** | Random Forest, SVM, XGBoost |
| **Imbalanced Data** | SMOTE (imbalanced-learn) |
| **Explainability** | SHAP |
| **Visualization** | matplotlib, seaborn, plotly |
| **Web Framework** | Streamlit |
| **Version Control** | Git, GitHub |
| **Notebooks** | Jupyter |

---

## 🎓 Learning Outcomes

By completing this project, you will learn:

✅ End-to-end machine learning pipeline (data → deployment)  
✅ Handling class imbalance with SMOTE  
✅ Hyperparameter tuning with GridSearchCV  
✅ Model evaluation beyond accuracy (F1, ROC-AUC, Precision-Recall)  
✅ Feature engineering and domain knowledge application (WHO standards)  
✅ Model interpretation with SHAP  
✅ Interactive web app deployment with Streamlit  
✅ Collaborative development with Git & GitHub  
✅ Real-world problem solving: predicting water safety  

---

## 🚀 Future Enhancements

- [ ] Add more classification models (Neural Networks, Gradient Boosting)
- [ ] Implement real-time IoT sensor data integration
- [ ] Add time-series analysis for temporal patterns
- [ ] Incorporate microbiological indicators (bacteria, viruses)
- [ ] Create REST API for model predictions (FastAPI)
- [ ] Add automated retraining pipeline
- [ ] Expand to multi-class classification (Excellent, Good, Fair, Poor water quality)

---

## 📝 References & Resources

- [Kaggle Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- [WHO Drinking-water Quality Standards](https://www.who.int/standards)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Official Docs](https://docs.streamlit.io/)
- [Scikit-learn Model Selection & Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 📄 License

This project is for **educational and research purposes**. The dataset is provided under the [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).

---

**Last Updated:** April 2026  
**Project Status:** ✅ Active Development
