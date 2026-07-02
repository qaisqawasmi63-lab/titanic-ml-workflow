# Titanic Dataset — Machine Learning Workflow

A full machine learning workflow applied to the Titanic dataset, covering data cleaning, feature engineering, and three modeling tasks: regression, classification, and clustering. Built to practice the complete pipeline end-to-end, not just model training in isolation.

## Dataset

The Titanic dataset (`titanic.csv`) contains passenger-level records with the following raw columns: `pclass`, `survival`, `sex`, `age`, `fare`, `embarked`.

(https://www.kaggle.com/datasets/fossouodonald/titaniccsv/data)

## Workflow

The project is split into five notebooks. The first two must run in order; the last three (Clustering, Classification, Regression) each branch independently off the feature-engineered dataset, so they can run in any order relative to each other.

| Order | Notebook | What it does |
|---|---|---|
| 1 | `1_Cleaning_Stage.ipynb` | Cleans raw data: fixes comma-formatted decimals in `age`/`fare`, imputes missing values with the median, maps `pclass`/`sex` to numeric, one-hot encodes `embarked` |
| 2 | `2_Feature_Engineering_Stage.ipynb` | Scales numeric features and creates an interaction feature (`pclass_sex_interaction`) |
| 3 | `3_Clustering_Analysis_Stage.ipynb` | Groups passengers using K-Means (optimal k chosen via the elbow method) on `age`, `fare`, and `pclass` |
| 4 | `4_Classification_Analysis_Stage.ipynb` | Predicts `survival` using Logistic Regression, Random Forest, and SVC |
| 5 | `5_Regression_Analysis_Stage.ipynb` | Predicts (scaled) `fare` using Linear, Ridge, Lasso, and Random Forest regressors |

Notebooks 3, 4, and 5 all read `titanic_engineered.csv` (the output of stage 2); Clustering additionally reads `titanic_preprocessed.csv` to map cluster results back to original-scale values.

## Results

### Clustering (grouping passengers)

K-Means clustering was applied to scaled `age`, `fare`, and `pclass`, with the optimal number of clusters selected using the elbow method (see `elbow_plot.png`). The resulting clusters separate passengers largely along class and fare lines, with distinct patterns in survival rate across groups.

### Classification (predicting Survival)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.780 | 0.791 | 0.624 | 0.697 |
| Random Forest | 0.770 | 0.713 | 0.729 | 0.721 |
| Support Vector Classifier | 0.756 | 0.743 | 0.612 | 0.671 |

Logistic Regression had the highest precision, while Random Forest had the best balance of precision and recall (highest F1 score). All three models scored in the mid-to-high 70s% accuracy range.

### Regression (predicting Fare)

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 0.460 | 0.403 |
| Ridge Regression | 0.461 | 0.402 |
| Lasso Regression | 0.494 | 0.359 |
| Random Forest Regressor | 0.683 | 0.114 |

Linear and Ridge Regression performed best, explaining ~40% of the variance in fare. Random Forest underperformed with default parameters, suggesting it would need tuning to compete here.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (models, scaling, train/test split, metrics)
- matplotlib, seaborn (elbow plot)

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Run the notebooks in numeric order (1 → 5) using Jupyter Notebook or JupyterLab.

## Files

- `1_Cleaning_Stage.ipynb`, `2_Feature_Engineering_Stage.ipynb`, `3_Clustering_Analysis_Stage.ipynb`, `4_Classification_Analysis_Stage.ipynb`, `5_Regression_Analysis_Stage.ipynb` — the five workflow notebooks, numbered in run order
- `titanic_preprocessed.csv`, `titanic_engineered.csv`, `titanic_clustered.csv` — intermediate outputs saved between stages
- `regression_results.csv`, `classification_results.csv`, `classification_report.txt` — model evaluation outputs
- `cluster_summary.csv` — clustered dataset with assigned cluster labels
- `elbow_plot.png` — elbow method plot used to choose k for K-Means

## Notes & Limitations

- The regression task predicts *scaled* fare, so MSE values are on a standardized scale, not raw dollar terms.
- Models were run with default/lightly-tuned hyperparameters; results could likely improve with tuning (especially the Random Forest models).
- `age` and `fare` in the raw data used commas instead of periods as decimal separators — this was caught and fixed during the cleaning stage.
