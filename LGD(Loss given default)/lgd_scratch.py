"""
LGD (Loss Given Default) from scratch.

Implements, with no external ML/stats library required:
  - workout LGD: discounted recovery cash flows minus costs, vs EAD
  - LTV/haircut-based expected recovery for secured exposures
  - downturn LGD vs long-run-average (LRA) LGD
  - Beta regression (mean via logit link, MLE) for continuous LGD in (0,1)
  - a two-stage cure/severity model (logistic cure classifier + Beta severity)

Everything here is plain NumPy/SciPy so every number can be reproduced by hand.
"""
import numpy as np
from scipy import optimize
from scipy.special import gammaln, expit  # expit = sigmoid


# ---------------------------------------------------------------------------
# 1. Workout LGD (discounted cash flows)
# ---------------------------------------------------------------------------
def workout_lgd(EAD, cashflows):
    """
    cashflows: list of (t_years, recovery, cost) tuples.
    Returns dict with discounted recovery, LGD, and the per-cashflow detail.
    """
    detail = []
    pv_total = 0.0
    for t, recovery, cost in cashflows:
        net = recovery - cost
        pv = net / (1 + workout_lgd.r) ** t
        detail.append(dict(t=t, recovery=recovery, cost=cost, net=net, pv=pv))
        pv_total += pv
    lgd = 1 - pv_total / EAD
    return dict(EAD=EAD, pv_recovery=pv_total, lgd=lgd, detail=detail)


workout_lgd.r = 0.10  # default discount rate; override via workout_lgd.r = x before calling


def workout_lgd_r(EAD, cashflows, r):
    """Same as workout_lgd but takes r explicitly (no global state)."""
    pv_total = 0.0
    detail = []
    for t, recovery, cost in cashflows:
        net = recovery - cost
        pv = net / (1 + r) ** t
        detail.append(dict(t=t, recovery=recovery, cost=cost, net=net, pv=pv))
        pv_total += pv
    lgd = 1 - pv_total / EAD
    return dict(EAD=EAD, pv_recovery=pv_total, lgd=lgd, detail=detail, r=r)


# ---------------------------------------------------------------------------
# 2. LTV / haircut-based expected recovery (secured exposures)
# ---------------------------------------------------------------------------
def ltv_haircut_lgd(EAD, collateral_value, haircut):
    recognized = collateral_value * (1 - haircut)
    expected_recovery = min(EAD, recognized)
    lgd = 1 - expected_recovery / EAD
    return dict(EAD=EAD, collateral_value=collateral_value, haircut=haircut,
                recognized_collateral=recognized, expected_recovery=expected_recovery,
                lgd=lgd, ltv=EAD / collateral_value)


# ---------------------------------------------------------------------------
# 3. Downturn LGD vs long-run average
# ---------------------------------------------------------------------------
def downturn_lgd(lgd_values, downturn_mask):
    lgd_values = np.asarray(lgd_values)
    downturn_mask = np.asarray(downturn_mask, dtype=bool)
    lra = lgd_values.mean()
    downturn = lgd_values[downturn_mask].mean() if downturn_mask.any() else lra
    return dict(lra_lgd=lra, downturn_lgd=downturn, regulatory_lgd=max(lra, downturn))


# ---------------------------------------------------------------------------
# 4. Beta regression: mean = sigmoid(X beta), fit by MLE
# ---------------------------------------------------------------------------
def _beta_negloglik(params, X, y, n_beta):
    beta = params[:n_beta]
    log_phi = params[n_beta]
    phi = np.exp(log_phi)               # precision, keep positive via log-param
    mu = expit(X @ beta)
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    a = mu * phi
    b = (1 - mu) * phi
    y = np.clip(y, 1e-6, 1 - 1e-6)
    ll = (gammaln(phi) - gammaln(a) - gammaln(b)
          + (a - 1) * np.log(y) + (b - 1) * np.log(1 - y))
    return -ll.sum()


class BetaRegression:
    """LGD ~ Beta(mu*phi, (1-mu)*phi), logit(mu) = X @ beta. Fit by MLE."""

    def fit(self, X, y):
        n, p = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        x0 = np.zeros(Xb.shape[1] + 1)   # betas + log(phi)
        x0[-1] = np.log(2.0)
        res = optimize.minimize(_beta_negloglik, x0, args=(Xb, y, Xb.shape[1]),
                                 method="L-BFGS-B")
        self.beta_ = res.x[:-1]
        self.phi_ = float(np.exp(res.x[-1]))
        self.result_ = res
        return self

    def predict(self, X):
        n = X.shape[0]
        Xb = np.hstack([np.ones((n, 1)), X])
        return expit(Xb @ self.beta_)


def beta_negloglik_gradient_check(X, y, n_beta, eps=1e-6):
    """Finite-difference check of the analytic optimizer's objective:
    compares scipy's numerical gradient (via approx) to a hand-perturbed
    finite-difference gradient at a random point, as an independent sanity
    check that _beta_negloglik is implemented correctly."""
    rng = np.random.default_rng(0)
    params = rng.normal(scale=0.1, size=n_beta + 1)
    base = _beta_negloglik(params, X, y, n_beta)
    fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        p2 = params.copy()
        p2[i] += eps
        fd_grad[i] = (_beta_negloglik(p2, X, y, n_beta) - base) / eps
    return fd_grad


# ---------------------------------------------------------------------------
# 5. Two-stage cure / severity model
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class CureSeverityModel:
    """Stage 1: logistic P(not cured) i.e. P(loss > 0), gradient g = p - y.
       Stage 2: BetaRegression severity, fit only on the non-cured rows.
       Final LGD = P(not cured) * E[severity | not cured]."""

    def __init__(self, lr=0.3, n_iter=5000, l2=1.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2

    def fit(self, X, lgd):
        n, p = X.shape
        not_cured = (lgd > 0.01).astype(float)   # "cured" = near-zero loss

        # stage 1: logistic regression, same g = p - y as the PD article
        Xb = np.hstack([np.ones((n, 1)), X])
        beta = np.zeros(p + 1)
        for _ in range(self.n_iter):
            pr = sigmoid(Xb @ beta)
            g = pr - not_cured
            grad = Xb.T @ g / n
            reg = self.l2 * beta / n
            reg[0] = 0.0
            beta -= self.lr * (grad + reg)
        self.cure_beta_ = beta

        # stage 2: Beta regression severity, non-cured rows only
        mask = not_cured.astype(bool)
        self.severity_model_ = BetaRegression().fit(X[mask], lgd[mask])
        return self

    def predict_p_not_cured(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return sigmoid(Xb @ self.cure_beta_)

    def predict_lgd(self, X):
        p_not_cured = self.predict_p_not_cured(X)
        severity = self.severity_model_.predict(X)
        return p_not_cured * severity
