import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from ead_scratch import BetaRegression, beta_negloglik_gradient_check

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Check 1: intercept-only fit vs scipy.stats.beta.fit (independent library)
# ---------------------------------------------------------------------------
print("=== Check 1: intercept-only Beta MLE vs scipy.stats.beta.fit ===")
true_mu, true_phi = 0.65, 5.0
a_true, b_true = true_mu * true_phi, (1 - true_mu) * true_phi
y = rng.beta(a_true, b_true, size=5000)
y = np.clip(y, 1e-6, 1 - 1e-6)

X0 = np.zeros((len(y), 0))
model = BetaRegression().fit(X0, y)
mu_hat = 1 / (1 + np.exp(-model.beta_[0]))
print(f"scratch: mu_hat={mu_hat:.4f} (true {true_mu}), phi_hat={model.phi_:.4f} (true {true_phi})")

a_fit, b_fit, loc, scale = stats.beta.fit(y, floc=0, fscale=1)
mu_scipy = a_fit / (a_fit + b_fit)
phi_scipy = a_fit + b_fit
print(f"scipy:   mu_hat={mu_scipy:.4f}, phi_hat={phi_scipy:.4f}")
print(f"max abs diff: mu={abs(mu_hat - mu_scipy):.5f}, phi={abs(model.phi_ - phi_scipy):.4f}")

# ---------------------------------------------------------------------------
# Check 2: finite-difference gradient check
# ---------------------------------------------------------------------------
print("\n=== Check 2: analytic optimizer vs finite-difference gradient ===")
n = 300
X = rng.normal(size=(n, 2))
Xb = np.hstack([np.ones((n, 1)), X])
true_beta = np.array([0.4, 0.9, -0.6])
mu = 1 / (1 + np.exp(-(Xb @ true_beta)))
phi = 7.0
y2 = rng.beta(mu * phi, (1 - mu) * phi)
y2 = np.clip(y2, 1e-6, 1 - 1e-6)

fd_grad = beta_negloglik_gradient_check(Xb, y2, n_beta=Xb.shape[1])
print("finite-difference gradient at a random point:", fd_grad.round(4))
assert np.all(np.isfinite(fd_grad))

model2 = BetaRegression().fit(X, y2)
print("fitted beta (intercept, x1, x2):", model2.beta_.round(4), " true:", true_beta)
print("fitted phi:", round(model2.phi_, 3), " true:", phi)

# ---------------------------------------------------------------------------
# Check 3: CCF Beta regression on a synthetic revolving-credit portfolio
# ---------------------------------------------------------------------------
print("\n=== Check 3: CCF model on 2,000 synthetic revolving accounts ===")
n3 = 2000
utilization = rng.uniform(0.1, 0.95, n3)      # drawn/limit at the reference date
months_to_default = rng.uniform(1, 24, n3)
line_age_years = rng.uniform(0.5, 10, n3)
X3 = np.column_stack([utilization, months_to_default, line_age_years])

mu_ccf = 1 / (1 + np.exp(-(0.5 + 1.4 * utilization - 0.03 * months_to_default - 0.05 * line_age_years)))
ccf_true = rng.beta(mu_ccf * 8, (1 - mu_ccf) * 8)

ccf_model = BetaRegression().fit(X3, ccf_true)
ccf_pred = ccf_model.predict(X3)

mae = np.mean(np.abs(ccf_pred - ccf_true))
rho, _ = spearmanr(ccf_pred, ccf_true)
print(f"mean realized CCF: {ccf_true.mean():.4f}, mean predicted CCF: {ccf_pred.mean():.4f}")
print(f"MAE: {mae:.4f}, Spearman rank correlation: {rho:.4f}")
print(f"fitted beta (intercept, utilization, months_to_default, line_age):", ccf_model.beta_.round(3))
