# 📊 Synthetic Data Platform

This sub-project generates a realistic synthetic dataset modeling early child stunting risk. The generation follows a Directed Acyclic Graph (DAG) for feature dependencies, applies health literature-based risk weights, injects noise, and performs rigorous validation.

## Directory Structure

```
synthetic_data/
│
├── config/                  # YAML configurations
│   ├── generator.yaml       # Pipeline parameters
│   ├── distributions.yaml   # Feature distributions
│   ├── relationships.yaml   # Inter-feature DAG
│   ├── risk_rules.yaml      # Stunting risk weights & thresholds
│   ├── validation.yaml      # Quality validation rules
│   └── export.yaml          # File outputs setting
│
├── src/                     # Source Code
│   ├── core/                # Core Orchestration (pipeline, config, builder)
│   ├── generators/          # Child/Parent/Household feature generators
│   ├── engines/             # DAG logic, Risk scoring, Noise addition
│   ├── validators/          # Data audit rules (ranges, dependencies)
│   ├── exporters/           # CSV / JSON / Metadata formatting
│   ├── reports/             # HTML QA auditing reports
│   └── utils/               # Logger, RNG Seed
│
├── output/                  # Generated CSV/JSON files (git-ignored)
└── reports/                 # Quality auditing HTML pages (git-ignored)
```

## Running the Platform

Ensure you are in the virtual environment. From the repository root directory, run:

```bash
python -m synthetic_data.src.main
```

Or from the `synthetic_data/` directory:

```bash
python src/main.py
```

### Outputs Generated in `output/` and `reports/`:
- `train.csv`, `validation.csv`, `test.csv`
- `metadata.json`, `statistics.json`, `relationships.json`, `risk_summary.json`
- `data_dictionary.csv`
- `quality_report.html`, `distribution_report.html`, `correlation_report.html`
