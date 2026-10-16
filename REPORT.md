# Model card: who to call for a term deposit

Author: Kubantov · Course: MOP 3231, Task 3 · Notebook: `kubantov_task3.ipynb`

## 1. Purpose
A bank phones clients to sell a term deposit. The call centre can phone only 10 % of the list in one wave. The model gives every client a score (higher = more likely to say "yes"), and the 10 % of clients with the highest scores are called. It is meant to rank clients **before** any call is made. It is not meant to give a probability for a single person.

## 2. Data
- `train.csv`: 32 940 clients, 19 usable features, target `y` (11.3 % "yes"). `test.csv`: 8 236 clients, no `y`.
- `duration` (call length) is **not used**: it is known only after the call and is missing in `test.csv`; using it would be target leakage. `id` is not used either.
- I split `train.csv` into a development part (26 352 rows) and a hold-out (6 588 rows), stratified by `y`, seed 3132. The hold-out was scored once, at the end.
- Missing values: `unknown` in categorical columns (e.g. `default` 20.9 %) and the value 999 in `pdays` (96.2 % of rows, meaning "not contacted before").

## 3. Pipeline
One scikit-learn `Pipeline`, saved in `model.joblib` (14.4 MB), accepting the raw columns of `test.csv` without `id`:
1. `pdays`: 999 is treated as missing, replaced by the median of the real values, plus a 0/1 column "was 999"; then scaled.
2. Other numeric columns: `StandardScaler`.
3. Categorical columns: one-hot encoding; `unknown` is kept as its own category (imputing it gave the same CV score); unseen categories are ignored.
4. `RandomForestClassifier`: 200 trees, `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`, `random_state=3132`.

All steps are fitted inside cross-validation folds. The final model is refitted on all of `train.csv`.

## 4. Metric and results
Because only 10 % can be called, I judge the **ranking**: ROC-AUC, average precision, and precision / recall / lift in the top 10 %. Accuracy is not used ("always no" already has 88.7 %).

| | ROC-AUC | Avg. precision | Precision@10 % |
|---|---|---|---|
| Majority class (5-fold CV) | 0.500 | 0.113 | 0.115 |
| Logistic regression, tuned (5-fold CV) | 0.793 ± 0.013 | 0.452 ± 0.028 | 0.500 ± 0.027 |
| **Random forest, tuned (5-fold CV)** | **0.801 ± 0.012** | **0.474 ± 0.026** | **0.522 ± 0.031** |
| Random forest, hold-out (once) | 0.807 | 0.460 | 0.501 |

On the hold-out, calling the top 10 % (659 of 6 588 clients, score cutoff 0.72) reaches 330 buyers: precision 0.50, recall 0.445, lift 4.45 (random calling would give about 74 buyers). 412 buyers are not called and 329 calls go to clients who say no.

The hold-out result agrees with cross-validation. The random forest was better than logistic regression in all 5 folds, but only by about 0.008 ROC-AUC, which is smaller than the fold-to-fold spread (about 0.012); I prefer it because it is best on every metric, not because it is proven much better. `class_weight="balanced"` did not improve the ranking.

## 5. Limits and ways it can fail
- **Random split flatters the numbers.** Clients from the same months and economic situation are in train and hold-out, and `month` and the macro-economic columns (`euribor3m`, `nr.employed`, `emp.var.rate`) identify the period. In use, the model scores future clients under different conditions, so real performance is probably lower.
- **Macro features dominate.** The strongest single features are macro-economic. If the economic situation moves outside the range seen in training, scores may become unreliable.
- **Moderate precision.** About half of the called clients still say no, and more than half of all buyers (55 %) are not reached.
- **Scores are not probabilities.** The forest uses balanced class weights, so `y_score` is for ranking only.
- **Personal attributes.** The model uses age, job, marital status and education. I did not check whether it treats groups differently, which could be a fairness risk.
- Months with few calls (e.g. December, 147 rows) give noisy estimates.

## 6. What to monitor after launch
- **Outcome:** the share of "yes" among called clients (compare with about 0.50 on the hold-out; a value well below it, for example under 0.40, is a warning) and the overall "yes" rate (about 11 %).
- **Inputs:** the distribution of `euribor3m`, `nr.employed` and `month`, the share of `unknown` and of `pdays = 999`, and the distribution of scores (cutoff about 0.72). Large shifts mean the training data no longer describes the clients.
- **Fairness:** precision and call rate by age group and job.
- **Lift check:** keep a small random group of clients to call, to measure the real lift against random calling.
- **Retraining:** refit on recent campaigns on a regular schedule, and whenever a monitored value leaves its normal range.