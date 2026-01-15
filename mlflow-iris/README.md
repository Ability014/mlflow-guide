# MLflow Iris Enterprise Reference Project

An enterprise-grade MLflow reference implementation demonstrating production ML lifecycle patterns, governance, and best practices.

## 🎯 Purpose

This repository serves as the **golden reference** for enterprise ML projects, implementing:

- ✅ **Experiment Tracking** — Parameters, metrics, artifacts with proper naming conventions
- ✅ **Model Packaging & Versioning** — Signatures, input examples, explicit dependencies
- ✅ **Model Registry & Promotion** — Dev → Staging → Production workflow
- ✅ **CI/CD Pipelines** — Automated testing, validation, deployment
- ✅ **Feature Versioning** — Lifecycle from local to Feature Store
- ✅ **Governance** — Unity Catalog integration, PR-based development

---

## 📁 Project Structure

```
mlflow-iris-enterprise/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # CI pipeline (tests, linting)
│       ├── cd-staging.yml            # Deploy to staging
│       └── cd-production.yml         # Deploy to production
├── configs/
│   ├── model_config.yaml             # Model hyperparameters
│   ├── feature_config.yaml           # Feature definitions (versioned)
│   └── environment.yaml              # Environment settings
├── docs/
│   ├── CONTRIBUTING.md               # Contribution guidelines
│   ├── MODEL_CARD.md                 # Model documentation
│   └── FEATURE_CATALOG.md            # Feature documentation
├── pipelines/
│   ├── training_pipeline.py          # Orchestrated training workflow
│   └── inference_pipeline.py         # Batch inference workflow
├── scripts/
│   ├── promote_model.py              # Model promotion script
│   ├── validate_model.py             # Model validation script
│   └── deploy_endpoint.py            # Endpoint deployment script
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py              # Data loading utilities
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_definitions.py    # Versioned feature definitions
│   │   └── feature_builder.py        # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py            # ⭐ Training entrypoint (required name)
│   │   ├── model_registry.py         # Registry operations
│   │   └── model_validator.py        # Model validation logic
│   ├── components/
│   │   ├── __init__.py
│   │   └── scoring.py                # Inference component
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # Configuration loader
│       ├── logging.py                # Logging utilities
│       └── mlflow_utils.py           # MLflow helper functions
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_features.py          # Feature engineering tests
│   │   ├── test_model.py             # Model training tests
│   │   └── test_data.py              # Data loading tests
│   └── integration/
│       ├── __init__.py
│       ├── test_training_pipeline.py # End-to-end training test
│       └── test_inference.py         # Inference pipeline test
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development dependencies
├── pyproject.toml                     # Project configuration
├── setup.py                           # Package setup
└── README.md                          # This file
```

---

## 🏗️ Architecture

### Model Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL PROMOTION WORKFLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│   │     DEV     │────▶│   STAGING   │────▶│ PRODUCTION  │                   │
│   │             │     │             │     │             │                   │
│   │ Experiments │     │ Validation  │     │  Approved   │                   │
│   │ Rapid iter. │     │ Testing     │     │  Governed   │                   │
│   └─────────────┘     └─────────────┘     └─────────────┘                   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│   MLflow Tracking     Model Registry      Model Serving                     │
│   - Parameters        - Versioning        - REST API                        │
│   - Metrics           - Lineage           - Autoscaling                     │
│   - Artifacts         - Approval          - Monitoring                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Feature Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE LIFECYCLE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────┐ │
│   │   LOCAL     │────▶│  VALIDATED  │────▶│ REFACTORED  │────▶│REGISTERED│ │
│   │             │     │             │     │             │     │          │ │
│   │ Defined in  │     │ Tested for  │     │ Generalized │     │ Feature  │ │
│   │ notebook    │     │ reuse       │     │ for teams   │     │ Store    │ │
│   └─────────────┘     └─────────────┘     └─────────────┘     └──────────┘ │
│                                                                   │         │
│                                                                   ▼         │
│                                                            Unity Catalog    │
│                                                            Governance       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Naming Conventions

### Experiment Naming
```
Pattern: /{team}/{project}/{model_name}

Examples:
- /data-science/iris/species_classifier
- /ml-platform/fraud/transaction_scorer
- /analytics/churn/customer_predictor
```

### Feature Naming
```
Pattern: {domain}_{feature_name}_v{version}

Examples:
- iris_sepal_length_cm_v1
- iris_petal_ratio_v2
- customer_lifetime_value_v3
```

### Model Naming (MLflow Registry)
```
Pattern: {catalog}.{schema}.{team}_{project}_{model_name}

Examples:
- prod_catalog.ml_models.ds_iris_species_classifier
- prod_catalog.ml_models.platform_fraud_scorer
```

### Run Naming
```
Pattern: {model_version}_{context}_{timestamp_or_build_id}

Examples:
- v1.2.0_full_20240115_143022
- v1.2.1_incremental_build_5678
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd mlflow-iris-enterprise

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### 2. Configure Environment

```bash
# Copy example config
cp configs/environment.yaml.example configs/environment.yaml

# Edit with your settings
# - MLFLOW_TRACKING_URI
# - UNITY_CATALOG_NAME
# - Team/project names
```

### 3. Run Training

```bash
# Run training pipeline
python -m src.models.train_model \
    --config configs/model_config.yaml \
    --experiment /data-science/iris/species_classifier \
    --context full \
    --version 1.0.0
```

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🔄 CI/CD Workflow

### Pull Request Workflow

```
1. Create feature branch from main
2. Make changes
3. Push and create PR
4. Automated checks run:
   - ✓ Linting (flake8, black)
   - ✓ Unit tests
   - ✓ Integration tests
   - ✓ Model validation
5. Code review required
6. Merge to main (protected)
7. CD pipeline triggers staging deployment
```

### Deployment Workflow

```yaml
# Trigger: PR merged to main
staging:
  - Run full test suite
  - Train model with staging config
  - Register model (staging alias)
  - Deploy to staging endpoint
  - Run validation tests

# Trigger: Manual approval
production:
  - Promote model (production alias)
  - Deploy to production endpoint
  - Run smoke tests
  - Enable monitoring
```

---

## 📊 MLflow Integration

### Experiment Tracking

Every training run logs:

| Category | Items Logged |
|----------|--------------|
| **Parameters** | Hyperparameters, data config, feature version |
| **Metrics** | Training & validation metrics (accuracy, F1, etc.) |
| **Artifacts** | Model, feature importance, confusion matrix |
| **Tags** | Model version, training context, build ID, git SHA |
| **Model** | Signature, input example, dependencies |

### Model Registry

Models are registered with:

| Attribute | Description |
|-----------|-------------|
| **Name** | `{catalog}.{schema}.{team}_{project}_{model}` |
| **Version** | Auto-incremented |
| **Aliases** | `dev`, `staging`, `production` |
| **Description** | Model card summary |
| **Tags** | Owner, use case, data lineage |

---

## 🧪 Testing Requirements

### Unit Tests (Required)
- Feature engineering logic
- Model training functions
- Data validation
- Configuration loading

### Integration Tests (Required)
- End-to-end training pipeline
- Model loading and inference
- Registry operations
- Feature store integration

---

## 📚 Documentation

- [Contributing Guide](docs/CONTRIBUTING.md)
- [Model Card](docs/MODEL_CARD.md)
- [Feature Catalog](docs/FEATURE_CATALOG.md)

---

## 🔐 Branch Protection

### Main Branch Rules
- ✓ Require PR for all changes
- ✓ Require at least 1 approval
- ✓ Require status checks to pass:
  - `test-unit`
  - `test-integration`
  - `lint`
  - `model-validation`
- ✓ Require branches to be up to date
- ✓ No direct pushes

---

## 📄 License

Internal use only. Contact ML Platform team for questions.