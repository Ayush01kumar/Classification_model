"""
PD (Probability of Default) from scratch.

Implements, with no external ML library required:
  - WOE (Weight of Evidence) and IV (Information Value) for a binned variable
  - a logistic-regression scorecard fit by gradient descent on g = p - y
  - PDO (points-to-double-odds) scaling from log-odds to a points score
  - Gini / AUC / KS for discriminatory power
  - PSI (Population Stability Index) for population drift
  - cumulative (lifetime) PD from a sequence of conditional annual PDs

Everything here is plain NumPy so every number can be reproduced by hand.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# 1. WOE / IV
# ---------------------------------------------------------------------------
def woe_iv(bin_labels, y, smoothing=0.0):
    """
    bin_labels : array-like of bin id per row (e.g. 0/1 for two bins)
    y          : 1/0 array, 1 = default (the "event")
    smoothing  : add this constant to every (events, non-events) cell
                 before computing WOE/IV (fixes the ln(x/0) problem when a
                 bin has zero events or zero non-events).

    Returns dict: bin -> (events, nonevents, woe), and total IV.
    """
    bin_labels = np.asarray(bin_labels)
    y = np.asarray(y)
    bins = sorted(set(bin_labels.tolist()))

    raw_events = {b: int(((bin_labels == b) & (y == 1)).sum()) for b in bins}
    raw_nonevents = {b: int(((bin_labels == b) & (y == 0)).sum()) for b in bins}

    tot_events = sum(raw_events.values()) + smoothing * len(bins)
    tot_nonevents = sum(raw_nonevents.values()) + smoothing * len(bins)

    out = {}
    iv_total = 0.0
    for b in bins:
        e = raw_events[b] + smoothing
        ne = raw_nonevents[b] + smoothing
        pct_e = e / tot_events
        pct_ne = ne / tot_nonevents
        woe = np.log(pct_ne / pct_e)
        iv_b = (pct_ne - pct_e) * woe
        out[b] = dict(events=raw_events[b], nonevents=raw_nonevents[b],
                      pct_events=pct_e, pct_nonevents=pct_ne, woe=woe, iv=iv_b)
        iv_total += iv_b
    return out, iv_total


# ---------------------------------------------------------------------------
# 2. Logistic regression by gradient descent (same g = p - y as the
#    Logistic Regression article), fit on WOE-transformed features.
# ---------------------------------------------------------------------------
class LogisticScorecard:
    def __init__(self, lr=0.1, n_iter=5000, l2=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2   # ridge penalty, same spirit as XGBoost's lambda; not applied to intercept

    def fit(self, X, y, verbose_first_step=False):
        n, p = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])   # intercept
        beta = np.zeros(p + 1)
        y = np.asarray(y, dtype=float)

        first_step = None
        for it in range(self.n_iter):
            F = Xb @ beta
            pr = sigmoid(F)
            g = pr - y                          # gradient of loss wrt F, per row
            grad = Xb.T @ g / n                 # gradient wrt beta
            reg = self.l2 * beta / n
            reg[0] = 0.0                        # never regularize the intercept
            grad = grad + reg
            if it == 0 and verbose_first_step:
                first_step = dict(p0=pr.copy(), g0=g.copy(), grad0=grad.copy())
            beta -= self.lr * grad
        self.beta_ = beta
        self.first_step_ = first_step
        return self

    def decision_function(self, X):
        n = X.shape[0]
        Xb = np.hstack([np.ones((n, 1)), X])
        return Xb @ self.beta_

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))


def points_scaling(base_score=600.0, base_odds=50.0, pdo=20.0):
    """Return (offset, factor) such that
       score = offset + factor * logit
       and increasing logit by ln(2) (odds double) moves score by pdo points."""
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return offset, factor


def to_score(logit, base_score=600.0, base_odds=50.0, pdo=20.0):
    offset, factor = points_scaling(base_score, base_odds, pdo)
    return offset + factor * logit


# ---------------------------------------------------------------------------
# 3. Discrimination: AUC, Gini, KS
# ---------------------------------------------------------------------------
def auc_gini_ks(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)          # riskiest (highest score) first
    y_sorted = y_true[order]

    n_pos = y_sorted.sum()                # defaults ("bad")
    n_neg = len(y_sorted) - n_pos         # non-defaults ("good")

    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg
    ks = float(np.max(np.abs(cum_pos - cum_neg))) * 100.0

    # AUC via rank-sum (Mann-Whitney), robust to ties
    ranks = y_score.argsort().argsort() + 1  # 1..n, ascending score -> rank
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    gini = 2 * auc - 1
    return dict(auc=auc, gini=gini, ks=ks, cum_pos=cum_pos, cum_neg=cum_neg)


# ---------------------------------------------------------------------------
# 4. PSI
# ---------------------------------------------------------------------------
def psi(expected_counts, actual_counts):
    """expected_counts, actual_counts: dict bin -> count (same bins)."""
    e_tot = sum(expected_counts.values())
    a_tot = sum(actual_counts.values())
    total = 0.0
    rows = []
    for b in expected_counts:
        e_pct = expected_counts[b] / e_tot
        a_pct = actual_counts.get(b, 0) / a_tot
        e_pct = max(e_pct, 1e-6)
        a_pct = max(a_pct, 1e-6)
        contrib = (a_pct - e_pct) * np.log(a_pct / e_pct)
        rows.append((b, e_pct, a_pct, contrib))
        total += contrib
    return total, rows


# ---------------------------------------------------------------------------
# 5. Cumulative (lifetime) PD from conditional annual PDs
# ---------------------------------------------------------------------------
def cumulative_pd(conditional_pds):
    survive = np.prod([1 - p for p in conditional_pds])
    return 1 - survive
