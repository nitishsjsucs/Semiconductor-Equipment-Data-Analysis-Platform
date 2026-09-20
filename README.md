# SECOM yield analysis

A scikit-learn study of the public SECOM semiconductor manufacturing dataset:
clean up 590 noisy sensor columns, rank which ones relate to pass/fail, train a
handful of classifiers on a badly imbalanced target, and run some unsupervised
anomaly detectors over the same data. Despite the repo name it is a local
analysis script plus a Streamlit viewer, not a platform.

## What's here

- `main.py` — CLI entry point (`--quick`, `--features N`, `--dashboard`).
- `src/pipeline.py` — the run: load, preprocess, select features, train, detect
  anomalies, write everything to `outputs/<timestamp>/`.
- `src/preprocessing.py` — drops columns with >50% missing and constant columns,
  median imputation, drops pairs correlated above 0.95, robust scaling.
- `src/feature_selection.py` — ranks features by F-test, mutual information,
  Random Forest importance, Spearman correlation and Gradient Boosting, then
  aggregates the ranks.
- `src/models.py` — Logistic Regression, KNN, SVM, AdaBoost, Gradient Boosting,
  Random Forest and an MLP, with SMOTE and balanced-accuracy model selection.
- `src/anomaly_detection.py` — Isolation Forest, LOF, One-Class SVM, Elliptic
  Envelope and a majority-vote ensemble.
- `src/ai_insights.py` — optional pass that sends the result summary to the
  OpenAI API (key read from `OPENAI_API_KEY`) for a written summary; without a
  key it falls back to canned text. The committed runs used the fallback
  (`ai_available: false` in `outputs/*/ai_insights.json`).
- `app.py` — Streamlit dashboard for browsing the data and results.
- `src/deep_learning.py` — a PyTorch autoencoder and classifier. Nothing imports
  it; it is not part of the pipeline.
- `notebooks/analysis.ipynb` — the same flow as a notebook, committed unexecuted.
- `outputs/<timestamp>/` — four saved runs (CSVs, plots, `summary.json`).

## Data

SECOM, from the UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/179/secom), committed under `secom/`:
1,567 samples, 590 sensor columns, 104 fail vs 1,463 pass (6.6% fail, 14:1
imbalance), ~4.5% of values missing. Preprocessing in the saved run cut 590
columns to 195 (24 too-missing, 265 constant, 106 correlated).

## Running it

```bash
pip install -r requirements.txt
python main.py                 # full run, writes outputs/<timestamp>/
python main.py --dashboard     # or: streamlit run app.py
```

## Results

From `outputs/20251228_155859/` (`model_results.csv`, `summary.json`), SMOTE on
the training split, scored on a held-out test set of 21 fails / 293 passes:

| Model | Balanced acc | Sensitivity | Specificity | ROC AUC |
|---|---|---|---|---|
| Logistic Regression (best) | 0.622 | 0.476 | 0.768 | 0.694 |
| KNN | 0.617 | 0.524 | 0.710 | 0.668 |
| Random Forest | 0.582 | 0.190 | 0.973 | 0.759 |

F1 on the fail class stays in the 0.19-0.24 range for every model. Anomaly
detection is similar: the best single detector (One-Class SVM) recalls 46% of
failures at 21% precision. Top-ranked sensors in that run: 129, 124, 59, 33, 64.

## Limitations

Small, heavily imbalanced dataset with anonymized sensors, so the models detect
roughly half the failures at best and there is no way to attribute a failure to
a physical cause. This is a class-scale exercise on a benchmark dataset, not a
tool for a real fab.
