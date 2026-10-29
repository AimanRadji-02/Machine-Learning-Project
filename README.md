# Machine Learning Projects: Nine Tasks, Four Datasets

A beginner-friendly Jupyter Notebook that implements nine classic machine-learning tasks with scikit-learn and TensorFlow/Keras. Every step is explained in plain English, and every task ends with a **"How to Explain This Model"** summary, so the notebook works as both a learning resource and a presentation aid.

The notebook inspects four datasets, decides which one is genuinely suitable for each task, and reports honest results, including cases where a model does **not** work.

## Tasks

| Level | Task | Algorithm | Dataset |
|---|---|---|---|
| Basic | 1. Data preprocessing | Imputer, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline | Stock prices |
| Basic | 2. Regression | `LinearRegression` (multiple features) | House prices |
| Basic | 3. Classification | `KNeighborsClassifier` | Iris |
| Intermediate | 4. Binary classification | `LogisticRegression` | Stock prices |
| Intermediate | 5. Decision trees | `DecisionTreeClassifier` with pruning | Iris |
| Intermediate | 6. Clustering | `KMeans` with elbow, silhouette and PCA | Iris (labels not used) |
| Advanced | 7. Ensemble learning | `RandomForestClassifier` with `GridSearchCV` | Stock prices |
| Advanced | 8. Support vector machines | `SVC` (linear vs RBF) | House prices |
| Advanced | 9. Neural network | Keras `Sequential` feed-forward network | Stock prices |

## Datasets

| File | Rows | Used for |
|---|---|---|
| `1) iris.csv` | 150 | KNN, decision tree, K-Means |
| `2) Stock Prices Data Set.csv` | 497,472 (505 companies, 2014 to 2017) | Preprocessing, logistic regression, random forest, neural network |
| `4) house Prediction Data Set.csv` | 506 | Linear regression, SVM |
| `Sentiment dataset.csv` | 732 | **Not used for modelling** (191 different labels, most very rare) |

Notes on the data:
- **House prices** has no header row; column names come from the public Boston housing documentation (an assumption, checked against value ranges). The `B` column is excluded because it is based on the racial composition of each area.
- **Stock prices** has no label. The target `target_up` (does tomorrow's close beat today's?) is created from the data using only today's and earlier information. The split is by date: train before 2017, test on 2017. The sample is 30 random companies plus 5 that contain missing values.
- **SVM target:** `expensive = 1` if `MEDV` is above the median. `MEDV` is removed from the features to avoid leakage.

## Selected results

All numbers come from running the notebook (seed 42), measured on held-out test data.

| Task | Result |
|---|---|
| 2. Linear regression | Test R² 0.689, MSE 22.78 (typical error about $4,800) |
| 3. KNN (K=7) | Test accuracy 0.967, macro F1 0.967 |
| 4. Logistic regression | Test accuracy 0.517, ROC-AUC 0.494 (about random) |
| 5. Decision tree (pruned) | Test accuracy 0.933, macro F1 0.933 |
| 6. K-Means (K=3) | Silhouette 0.457, adjusted Rand index vs species 0.62 |
| 7. Random forest (tuned) | Test accuracy 0.532, ROC-AUC 0.533 |
| 8. SVM | Linear: accuracy 0.912, AUC 0.964. RBF: accuracy 0.892, AUC 0.975 |
| 9. Neural network | Run the notebook to generate (see note below) |

### What the results teach

- Iris and the house data have strong feature-target relationships, so simple models score well.
- Predicting next-day stock direction from simple price features is essentially at chance level (always guessing "up" scores 0.524). An unrestricted random forest reached 1.000 training accuracy but only 0.505 on test data, a clear example of overfitting.
- The Iris and house test sets are small (30 and 102 rows), so small score differences between models are not strong evidence.

## Good-practice highlights

- Preprocessing fitted on training data only, inside scikit-learn pipelines, to prevent data leakage.
- Hyperparameters chosen with cross-validation on the training set, never on the test set.
- Time-ordered split and `TimeSeriesSplit` cross-validation for the stock data.
- Results compared against a simple baseline, not just reported in isolation.
- Appropriate metrics for each task: MSE and R², accuracy, precision, recall, F1, ROC-AUC, confusion matrices, inertia and silhouette.

## Getting started

1. Clone the repository and put the four CSV files in a folder named `datasets` next to the notebook (or change `DATA_FOLDER` in the configuration cell).
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
   TensorFlow is only needed for Task 9.
3. Open `machine_learning_projects.ipynb` in Jupyter, JupyterLab or Google Colab and run the cells from top to bottom.

## Repository structure

```
.
├── machine_learning_projects.ipynb
├── requirements.txt
├── README.md
└── datasets/            # place the four CSV files here
```

## Limitations

- The neural network section (Task 9) was written but **not executed** when the notebook was prepared, because TensorFlow was unavailable in that environment. Run it yourself and read your own output before drawing conclusions.
- The stock features are deliberately simple; this is a learning project, not a trading system.
- Results can vary slightly with different library versions.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, TensorFlow/Keras.
