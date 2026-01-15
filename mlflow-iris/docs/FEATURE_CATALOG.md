# Feature Catalog

## Overview

This document catalogs all features available for the Iris Species Classifier model.

**Naming Convention:** `{domain}_{feature_name}_v{version}`

## Feature Lifecycle

| Status | Description |
|--------|-------------|
| `local` | Defined in notebook/local code, not validated |
| `validated` | Tested for correctness and reuse potential |
| `refactored` | Generalized for team-wide use |
| `registered` | Published to Feature Store, governed via Unity Catalog |

---

## Base Features

### iris_sepal_length_cm_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_sepal_length_cm_v1` |
| **Description** | Sepal length in centimeters |
| **Type** | `float64` |
| **Source** | `sepal_length` column |
| **Status** | `registered` |
| **Range** | 4.3 - 7.9 cm |

### iris_sepal_width_cm_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_sepal_width_cm_v1` |
| **Description** | Sepal width in centimeters |
| **Type** | `float64` |
| **Source** | `sepal_width` column |
| **Status** | `registered` |
| **Range** | 2.0 - 4.4 cm |

### iris_petal_length_cm_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_petal_length_cm_v1` |
| **Description** | Petal length in centimeters |
| **Type** | `float64` |
| **Source** | `petal_length` column |
| **Status** | `registered` |
| **Range** | 1.0 - 6.9 cm |

### iris_petal_width_cm_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_petal_width_cm_v1` |
| **Description** | Petal width in centimeters |
| **Type** | `float64` |
| **Source** | `petal_width` column |
| **Status** | `registered` |
| **Range** | 0.1 - 2.5 cm |

---

## Derived Features

### iris_petal_ratio_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_petal_ratio_v1` |
| **Description** | Ratio of petal length to width |
| **Type** | `float64` |
| **Derivation** | `petal_length / petal_width` |
| **Status** | `validated` |
| **Notes** | Good discriminator for setosa |

### iris_sepal_ratio_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_sepal_ratio_v1` |
| **Description** | Ratio of sepal length to width |
| **Type** | `float64` |
| **Derivation** | `sepal_length / sepal_width` |
| **Status** | `validated` |

### iris_petal_area_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_petal_area_v1` |
| **Description** | Approximate petal area |
| **Type** | `float64` |
| **Derivation** | `petal_length * petal_width` |
| **Status** | `local` |

### iris_sepal_area_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_sepal_area_v1` |
| **Description** | Approximate sepal area |
| **Type** | `float64` |
| **Derivation** | `sepal_length * sepal_width` |
| **Status** | `local` |

---

## Feature Groups

### base_v1

Standard 4-feature set for baseline models.
```yaml
features:
  - iris_sepal_length_cm_v1
  - iris_sepal_width_cm_v1
  - iris_petal_length_cm_v1
  - iris_petal_width_cm_v1
```

### enhanced_v1

Base features plus ratio features.
```yaml
features:
  - iris_sepal_length_cm_v1
  - iris_sepal_width_cm_v1
  - iris_petal_length_cm_v1
  - iris_petal_width_cm_v1
  - iris_petal_ratio_v1
  - iris_sepal_ratio_v1
```

### full_v1

All available features including area calculations.
```yaml
features:
  - iris_sepal_length_cm_v1
  - iris_sepal_width_cm_v1
  - iris_petal_length_cm_v1
  - iris_petal_width_cm_v1
  - iris_petal_ratio_v1
  - iris_sepal_ratio_v1
  - iris_petal_area_v1
  - iris_sepal_area_v1
```

---

## Target Variable

### iris_species_encoded_v1

| Attribute | Value |
|-----------|-------|
| **Name** | `iris_species_encoded_v1` |
| **Description** | Encoded species label |
| **Type** | `int64` |
| **Classes** | 0=setosa, 1=versicolor, 2=virginica |

---

## Feature Store Location

| Attribute | Value |
|-----------|-------|
| **Catalog** | `prod_catalog` |
| **Schema** | `ml_features` |
| **Table** | `iris_features` |
| **Primary Key** | `sample_id` |
| **Timestamp** | `event_timestamp` |

---

## Versioning Guidelines

1. **Increment version** when:
   - Feature computation logic changes
   - Data type changes
   - Semantics change

2. **Create new feature** when:
   - Adding completely new computation
   - Significantly different meaning

3. **Deprecate** (don't delete) when:
   - Feature no longer recommended
   - Better alternative exists