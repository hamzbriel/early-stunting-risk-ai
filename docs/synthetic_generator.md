# Synthetic Data Generator

## Overview

The Synthetic Data Platform generates realistic child stunting risk datasets using **probabilistic modeling**, **Directed Acyclic Graph (DAG)** relationships, and **domain-driven risk rules**. This approach produces data that reflects real-world patterns described in public health literature without requiring access to sensitive medical records.

> 10.000 synthetic records are generated, representing children under 5 years old across diverse socioeconomic and geographic backgrounds.

## Pipeline Architecture

```
Configuration (YAML)
    │
    ▼
┌─────────────────────────────────────┐
│ Feature Generation                  │
│ - Marginal distributions per config │
│ - Uniform, Normal, Bernoulli, etc.  │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│ Relationship Engine                 │
│ - Apply DAG edges                   │
│ - Adjust probabilities/categories   │
│ - Enforce feature dependencies      │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│ Risk Engine                         │
│ - Weighted scoring (10 features)    │
│ - Interaction effects               │
│ - Risk level thresholds             │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│ Noise Injection + Validation        │
│ - Gaussian noise (std=3.5)          │
│ - Range & category checks           │
│ - Dependency consistency checks     │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│ Export (Train/Validation/Test)      │
│ - 70% / 15% / 15% split            │
│ - Stratified by risk_level         │
└─────────────────────────────────────┘
```

## Configuration Files

All generation parameters are defined in YAML configuration files under `synthetic_data/config/`:

| File | Purpose |
|------|---------|
| `generator.yaml` | Master config - sample size, seed, split ratios |
| `distributions.yaml` | Marginal probability distributions for each feature |
| `relationships.yaml` | DAG edges defining inter-feature dependencies |
| `risk_rules.yaml` | Weighted scoring, interactions, and thresholds |
| `validation.yaml` | Post-generation validation rules and bounds |
| `export.yaml` | Output file names, column order, report settings |

## Feature Generation

### Distribution Types

The generator supports four distribution types, configured in `distributions.yaml`:

| Type | Features | Parameters |
|------|----------|------------|
| `normal` (continuous) | `birth_weight`, `birth_length`, `mother_age` | `mean`, `std`, `clip_min`, `clip_max` |
| `uniform_int` | `age_month` | `min`, `max` |
| `categorical` | `gender`, `mother_education`, `family_income`, `sanitation`, etc. | `categories`, `probabilities` |
| `bernoulli` (binary) | `mother_working`, `clean_water`, `electricity`, `exclusive_breastfeeding` | `base_probability` |

### Example - Birth Weight Distribution

```yaml
birth_weight:
  distribution: normal
  mean: 3.1
  std: 0.45
  clip_min: 1.5
  clip_max: 5.0
  description: "Birth weight in kilograms"
```

## Relationship Engine (DAG)

Features are **not** generated in isolation. The Relationship Engine applies a Directed Acyclic Graph to model real-world dependencies:

### Socioeconomic Chain

```
Mother Education ─────────────────────────────────┐
    +                                             ├──→ Family Income
Father Education ─────────────────────────────────┘
    │
    ├──→ Mother Working (education ↑ → employment ↑)
    │
    └──→ Protein / Vitamin Intake (via income + education)
```

### WASH Chain (Water, Sanitation, Hygiene)

```
Family Income + House Density → Sanitation
Sanitation + Family Income → Clean Water
Clean Water + Sanitation + Immunization → Diarrhea History
```

### Healthcare Chain

```
Family Income + Mother Education → Healthcare Access
Healthcare Access → Immunization
```

### Biological Correlation

```
Birth Weight ←→ Birth Length (correlation = 0.75)
```

### Relationship Types

Two types of relationship rules are supported:

- **`probability_modifier`**: Adjusts the base probability of a binary feature (e.g., `mother_working` probability increases by +0.25 if `mother_education` is "higher").
- **`category_shift`**: Shifts categorical probability distributions toward a target category with a given strength (e.g., shifts `sanitation` toward "good" if income is "high").

## Risk Engine

The Risk Engine calculates the `risk_score` (0–100) and assigns `risk_level` using a multi-step process.

### Step 1: Weighted Score

Each of 10 features is assigned a weight based on its contribution to stunting risk in public health literature:

| Feature | Weight | Rationale |
|---------|--------|-----------|
| `birth_weight` | **0.18** | Low birth weight is the strongest single predictor |
| `protein_intake` | 0.14 | Chronic protein deficiency is directly causal |
| `family_income` | 0.12 | Socioeconomic status underlies many risk factors |
| `sanitation` | 0.11 | Poor sanitation → infection → stunting pathway |
| `mother_education` | 0.10 | Education affects feeding practices and care |
| `clean_water` | 0.09 | Waterborne disease pathway |
| `diarrhea_history` | 0.08 | Repeated diarrhea directly impairs growth |
| `immunization` | 0.07 | Prevents infections that can cause stunting |
| `birth_length` | 0.06 | Correlated with birth weight |
| `healthcare_access` | 0.05 | Preventive and curative care access |

### Step 2: Feature Scoring

Each feature value is mapped to a 0–100 risk scale:

- **Continuous inverse** (birth_weight, birth_length): Lower values → higher risk score (linear interpolation)
- **Categorical** (protein_intake, family_income, etc.): Each category maps to a risk score
- **Binary** (clean_water, diarrhea_history): 0 (absent/negative) → higher risk

### Step 3: Interaction Effects

Synergistic combinations amplify risk, while protective combinations reduce it:

| Type | Condition | Effect |
|------|-----------|--------|
| Synergistic | `very_low` income + `poor` sanitation | **+8.0** |
| Synergistic | birth weight < 2.5 kg + `low` protein intake | **+7.0** |
| Synergistic | no clean water + diarrhea history | **+6.0** |
| Protective | `higher` education + `high` income | **-7.0** |
| Protective | `good` sanitation + clean water + complete immunization | **-6.0** |

### Step 4: Noise & Thresholds

1. Add Gaussian noise (std = 3.5) to simulate real-world measurement uncertainty
2. Clip final score to [0, 100]
3. Assign risk level:

| Risk Score | Label |
|------------|-------|
| 0 – 35 | **Low** |
| 36 – 65 | **Medium** |
| 66 – 100 | **High** |

## Validation

Post-generation validation checks include:

- **Range validation**: Numeric fields within biological bounds
- **Category validation**: Categorical fields contain only allowed values
- **Binary validation**: Binary fields are exactly 0 or 1
- **Dependency consistency**: Logical checks (e.g., high-income households should rarely have poor sanitation)
- **Duplicate detection**: Warns if >1% exact duplicates
- **Distribution validation**: Checks risk_level proportions match expected (Low: 35%, Medium: 40%, High: 25%)

## Usage

```bash
cd synthetic_data
python src/main.py
```

Output is saved to `synthetic_data/output/` with the following files:

| File | Description |
|------|-------------|
| `train.csv` | 70% of records - model training |
| `validation.csv` | 15% - model validation |
| `test.csv` | 15% - model testing/evaluation |
| `metadata.json` | Dataset generation metadata |
| `statistics.json` | Feature statistics (mean, std, etc.) |
| `data_dictionary.csv` | Column definitions and allowed values |

## Extending the Generator

### Adding a New Feature

1. Add distribution config in `distributions.yaml`
2. (Optional) Add relationship edges in `relationships.yaml`
3. (Optional) Add risk scoring in `risk_rules.yaml`
4. Add validation rules in `validation.yaml`
5. Add to column order in `export.yaml`

### Modifying Risk Rules

All risk parameters are in `risk_rules.yaml` - adjust weights, scoring mappings, interaction effects, or thresholds without touching Python code.

### Reproducibility

The generator is fully reproducible using `seed: 42` in `generator.yaml`. Changing the seed produces a different but statistically equivalent dataset.
