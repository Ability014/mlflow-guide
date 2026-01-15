# Model Card: Iris Species Classifier

## Model Details

| Attribute | Value |
|-----------|-------|
| **Model Name** | iris_species_classifier |
| **Version** | 1.0.0 |
| **Type** | Multi-class Classification |
| **Framework** | scikit-learn |
| **Owner** | ML Platform Team |

## Intended Use

### Primary Use Case
Classify iris flowers into three species (setosa, versicolor, virginica) based on sepal and petal measurements.

### Intended Users
- Data scientists for reference implementation
- ML engineers for deployment patterns
- Analysts for iris species identification

### Out-of-Scope Uses
- Production classification of non-iris flowers
- Use without proper input validation

## Training Data

| Attribute | Value |
|-----------|-------|
| **Dataset** | UCI Iris / sklearn |
| **Samples** | 150 |
| **Classes** | 3 (balanced) |
| **Features** | 4 numeric |

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | >0.95 |
| Macro F1 | >0.95 |
| Inference Time | <10ms |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Setosa | ~1.0 | ~1.0 | ~1.0 |
| Versicolor | ~0.94 | ~0.94 | ~0.94 |
| Virginica | ~0.94 | ~0.94 | ~0.94 |

## Limitations

- Limited to 4 input features
- May struggle with edge cases between versicolor/virginica
- Trained on limited sample size

## Ethical Considerations

- No known ethical concerns for iris classification
- Model does not process personal data

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01 | Initial release |