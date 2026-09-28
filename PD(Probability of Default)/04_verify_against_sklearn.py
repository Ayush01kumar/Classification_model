import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pd_scratch import LogisticScorecard, auc_gini_ks, psi

rng = 42
X, y = make_classification(n_samples=2000, n_features=8, n_informative=5,
                            n_redundant=1, weights=[0.85, 0.15],  # imbalanced, like real default rates
                            class_sep=1.2, random_state=rng)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=rng)

# standardize (mean 0, std 1) so gradient descent behaves; sklearn does this internally
# via its optimizer, but we standardize explicitly for the scratch model.
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
Xtr_s = (Xtr - mu) / sd
Xte_s = (Xte - mu) / sd

print("=== Scratch logistic regression (gradient descent, l2=0, i.e. unregularized) ===")
scratch = LogisticScorecard(lr=0.3, n_iter=8000, l2=0.0).fit(Xtr_s, ytr)
print("scratch beta (intercept + 8 coefs):\n", scratch.beta_)

print("\n=== sklearn LogisticRegression (no penalty, to compare apples to apples) ===")
sk = LogisticRegression(penalty=None, max_iter=5000)
sk.fit(Xtr_s, ytr)
print("sklearn intercept:", sk.intercept_)
print("sklearn coef:", sk.coef_)

diff = np.abs(np.concatenate([sk.intercept_, sk.coef_.ravel()]) - scratch.beta_)
print("\nmax abs coefficient difference:", diff.max())

print("\n=== Test-set discrimination: scratch vs sklearn ===")
p_scratch = scratch.predict_proba(Xte_s)
p_sk = sk.predict_proba(Xte_s)[:, 1]

res_scratch = auc_gini_ks(yte, p_scratch)
res_sk = auc_gini_ks(yte, p_sk)
print("scratch  AUC/Gini/KS:", res_scratch["auc"], res_scratch["gini"], res_scratch["ks"])
print("sklearn  AUC/Gini/KS:", res_sk["auc"], res_sk["gini"], res_sk["ks"])
print("sklearn roc_auc_score (library check):", roc_auc_score(yte, p_sk))
print("scratch predicted-prob vs sklearn predicted-prob, max abs diff:",
      np.abs(p_scratch - p_sk).max())

print("\n=== PSI sanity check: same distribution vs itself should be ~0 ===")
bins = np.digitize(p_sk, np.quantile(p_sk, [0.25, 0.5, 0.75]))
exp_counts = {b: int((bins == b).sum()) for b in set(bins.tolist())}
same_psi, _ = psi(exp_counts, exp_counts)
print("PSI of a distribution against itself:", same_psi)
