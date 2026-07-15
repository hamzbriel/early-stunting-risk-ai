# Machine Learning Model

## Overview

The ML pipeline compares multiple classification algorithms and selects the best-performing model for deployment. **CatBoost** emerged as the top performer, achieving **90.6% test accuracy** with strong precision and recall across all three risk classes.

## Models Compared

| Model | Cross-Val Accuracy | Precision | Recall | F1-Score |
|-------|-------------------|-----------|--------|----------|
| **CatBoost** | **88.91%** | 0.889 | 0.889 | **0.889** |
| Logistic Regression | 88.79% | 0.888 | 0.888 | 0.887 |
| LightGBM | 88.49% | 0.885 | 0.885 | 0.884 |
| XGBoost | 88.29% | 0.883 | 0.883 | 0.882 |
| Random Forest | 87.70% | 0.878 | 0.877 | 0.875 |
| Extra Trees | 86.94% | 0.870 | 0.869 | 0.868 |

*Cross-validation: 5-fold stratified, scoring metric: f1_weighted*

## Best Model: CatBoost

**CatBoost** was selected for deployment because:

- **Native categorical handling** - no manual encoding required during training
- **High robustness** - lowest accuracy std across folds (±0.006)
- **Best F1-score** - 0.889, indicating balanced precision and recall
- **Fast inference** - suitable for real-time prediction API

### Test Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | **0.906** |
| Balanced Accuracy | 0.876 |
| Precision | 0.906 |
| Recall | 0.906 |
| F1-Score | 0.906 |
| Matthews Correlation (MCC) | 0.828 |

> 1,500 test samples; model pipeline loaded from `model/trained_models/best_pipeline.pkl`

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model type | Gradient Boosting (CatBoost) |
| Training date | 11 Juli 2026 |
| Training samples | 8,500 |
| Test samples | 1,500 |
| Original features | 20 |
| Processed features | 42 |
| Cross-validation folds | 5 |
| Random seed | 42 |
| Target | `risk_level` (Low / Medium / High) |

## Feature Preprocessing

The model pipeline (`best_pipeline.pkl`) includes:

1. **ColumnTransformer**:
   - Numerical features → StandardScaler
   - Categorical features → OneHotEncoder
2. **CatBoostClassifier** as the final estimator

## Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to make predictions interpretable. All SHAP computation is done during the **training phase** (not at runtime) using `TreeExplainer` for the CatBoost model.

### Global Feature Importance

Top features by mean absolute SHAP value (300 samples analyzed):

| Rank | Feature | Mean SHAP |
|------|---------|-----------|
| 1 | `clean_water` | 0.765 |
| 2 | `diarrhea_history` | 0.702 |
| 3 | `protein_intake_low` | 0.495 |
| 4 | `sanitation_good` | 0.468 |
| 5 | `protein_intake_high` | 0.389 |
| 6 | `birth_weight` | 0.385 |
| 7 | `sanitation_poor` | 0.363 |
| 8 | `immunization_complete` | 0.327 |
| 9 | `family_income_high` | 0.238 |
| 10 | `family_income_very_low` | 0.228 |

### How SHAP Values Are Used

SHAP analysis confirms that the model's decision-making aligns with known public health risk factors:

- **Clean water access** is the single most important feature - reflecting the WASH (Water, Sanitation, Hygiene) pathway to stunting
- **Diarrhea history** and **protein intake** rank highly, consistent with the infection-malnutrition cycle
- **Birth weight** remains a strong predictor, as expected from clinical literature
- Features with low SHAP importance (e.g., `gender`, `father_education`) contribute minimally to predictions, which aligns with domain knowledge

### Accessing Explanations

Pre-computed SHAP results are exposed via two API endpoints:

- `GET /feature-importance` - Global feature importance scores
- `GET /explanation-summary` - SHAP summary with interpretation guidance

For individual predictions, the `POST /predict` response includes `top_risk_factors` based on global SHAP importance, providing users with actionable insights about the most influential features for their specific prediction.

## Model Artifacts

All artifacts are stored under `model/`:

```
model/
├── trained_models/
│   └── best_pipeline.pkl         ← CatBoost pipeline (loaded at runtime)
├── artifacts/
│   ├── label_encoder.pkl         ← LabelEncoder for risk_level
│   ├── feature_names.json        ← 20 original column names
│   └── training_config.json      ← Training metadata
├── metrics/
│   ├── metrics.json              ← Test set metrics
│   ├── cross_validation.json     ← 5-fold CV results (all models)
│   └── model_info.json           ← CatBoost metadata
└── explainability/
    ├── feature_importance.csv    ← SHAP values per feature
    └── explanation_summary.json  ← SHAP global summary
```
