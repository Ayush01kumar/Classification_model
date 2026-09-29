"""
EAD (Exposure at Default) from scratch.

Implements, with no external ML/stats library required:
  - CCF (Credit Conversion Factor) from a fixed-horizon historical cohort
  - EAD forecast for a currently-performing revolving facility
  - the Basel EAD floor (EAD can never be estimated below the current draw)
  - downturn CCF vs long-run-average CCF
  - Beta regression for CCF (same reparameterized-Beta MLE as the LGD article)
  - SA-CCR: EAD for a derivatives/counterparty-credit-risk exposure

Everything here is plain NumPy/SciPy so every number can be reproduced by hand.
"""
import numpy as np
from scipy import optimize
from scipy.special import gammaln, expit  # expit = sigmoid


# ---------------------------------------------------------------------------
# 1. CCF from a fixed-horizon historical cohort, and EAD forecast
# ---------------------------------------------------------------------------
def ccf_from_cohort(drawn_ref, limit_ref, drawn_default):
    """Realized CCF for one historical defaulted account, observed at a fixed
    horizon (e.g. 12 months) before default."""
    undrawn_ref = limit_ref - drawn_ref
    if undrawn_ref <= 0:
        return 0.0  # already fully drawn at the reference date; nothing left to convert
    ccf = (drawn_default - drawn_ref) / undrawn_ref
    return ccf


def ead_forecast(drawn_now, limit_now, ccf, floor_at_drawn=True):
    """Apply an estimated CCF to a currently-performing account to forecast
    its EAD if it were to default. Basel's EAD floor: EAD can never be
    estimated below the currently drawn amount."""
    undrawn_now = max(limit_now - drawn_now, 0.0)
    ead = drawn_now + ccf * undrawn_now
    if floor_at_drawn:
        ead = max(ead, drawn_now)
    return ead


# ---------------------------------------------------------------------------
# 2. Downturn CCF vs long-run average
# ---------------------------------------------------------------------------
def downturn_ccf(ccf_values, downturn_mask):
    ccf_values = np.asarray(ccf_values)
    downturn_mask = np.asarray(downturn_mask, dtype=bool)
    lra = ccf_values.mean()
    downturn = ccf_values[downturn_mask].mean() if downturn_mask.any() else lra
    return dict(lra_ccf=lra, downturn_ccf=downturn, regulatory_ccf=max(lra, downturn))


# ---------------------------------------------------------------------------
# 3. Beta regression for CCF: mean = sigmoid(X beta), fit by MLE
#    (same construction as LGD's BetaRegression - CCF is bounded like LGD)
# ---------------------------------------------------------------------------
def _beta_negloglik(params, X, y, n_beta):
    beta = params[:n_beta]
    log_phi = params[n_beta]
    phi = np.exp(log_phi)
    mu = expit(X @ beta)
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    a, b = mu * phi, (1 - mu) * phi
    y = np.clip(y, 1e-6, 1 - 1e-6)
    ll = (gammaln(phi) - gammaln(a) - gammaln(b)
          + (a - 1) * np.log(y) + (b - 1) * np.log(1 - y))
    return -ll.sum()


class BetaRegression:
    """CCF ~ Beta(mu*phi, (1-mu)*phi), logit(mu) = X @ beta. Fit by MLE."""

    def fit(self, X, y):
        n, p = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        x0 = np.zeros(Xb.shape[1] + 1)
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


def beta_negloglik_gradient_check(Xb, y, n_beta, eps=1e-6):
    """Finite-difference check of _beta_negloglik at a random point."""
    rng = np.random.default_rng(0)
    params = rng.normal(scale=0.1, size=n_beta + 1)
    base = _beta_negloglik(params, Xb, y, n_beta)
    fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        p2 = params.copy()
        p2[i] += eps
        fd_grad[i] = (_beta_negloglik(p2, Xb, y, n_beta) - base) / eps
    return fd_grad


# ---------------------------------------------------------------------------
# 4. SA-CCR: EAD for a derivative / counterparty-credit-risk exposure
# ---------------------------------------------------------------------------
def sa_ccr_ead(replacement_cost, potential_future_exposure, alpha=1.4):
    """Basel's Standardized Approach for Counterparty Credit Risk.
    EAD = alpha * (RC + PFE)."""
    rc = max(replacement_cost, 0.0)
    ead = alpha * (rc + potential_future_exposure)
    return dict(RC=rc, PFE=potential_future_exposure, alpha=alpha, EAD=ead)


def sa_ccr_pfe(notional, supervisory_factor, maturity_factor=1.0,
               multiplier=1.0):
    """Simplified single-trade PFE add-on:
    AddOn = notional * supervisory_factor * maturity_factor
    PFE = multiplier * AddOn (multiplier=1 with no netting benefit)."""
    add_on = notional * supervisory_factor * maturity_factor
    pfe = multiplier * add_on
    return dict(add_on=add_on, pfe=pfe)
