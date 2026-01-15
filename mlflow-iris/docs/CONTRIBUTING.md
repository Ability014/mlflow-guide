# Contributing Guide

## Development Workflow

### 1. Branch Protection
- `main` branch is protected
- All changes require PR
- Required checks must pass

### 2. Creating a PR
```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes
git add .
git commit -m "feat: description"

# Push and create PR
git push origin feature/your-feature
```

### 3. Required Checks
- ✓ `lint` - Code formatting (black, isort, flake8)
- ✓ `test-unit` - Unit tests pass
- ✓ `test-integration` - Integration tests pass
- ✓ `model-validation` - Model meets thresholds

### 4. Code Review
- At least 1 approval required
- Address all comments
- Keep PRs focused and small

## Code Standards

### Formatting
```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src
```

## Model Changes

When modifying model code:
1. Update version in `configs/model_config.yaml`
2. Update MODEL_CARD.md if behavior changes
3. Ensure all tests pass
4. Validate model meets thresholds

## Feature Changes

When adding/modifying features:
1. Update `configs/feature_config.yaml`
2. Increment feature version
3. Update FEATURE_CATALOG.md
4. Add tests for new features