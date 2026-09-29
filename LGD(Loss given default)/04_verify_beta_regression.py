import numpy as np
from scipy import stats
from lgd_scratch import BetaRegression, beta_negloglik_gradient_check, CureSeverityModel

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Check 1: intercept-only fit vs scipy.stats.beta.fit (independent library)
# ---------------------------------------------------------------------------
print("=== Check 1: intercept-only Beta MLE vs scipy.stats.beta.fit ===")
true_mu, true_phi = 0.35, 6.0
a_true, b_true = true_mu * true_phi, (1 - true_mu) * true_phi
y = rng.beta(a_true, b_true, size=5000)
y = np.clip(y, 1e-6, 1 - 1e-6)

X0 = np.zeros((len(y), 0))  # no covariates, intercept only
model = BetaRegression().fit(X0, y)
mu_hat = 1 / (1 + np.exp(-model.beta_[0]))
print(f"scratch: mu_hat={mu_hat:.4f} (true {true_mu}), phi_hat={model.phi_:.4f} (true {true_phi})")

a_fit, b_fit, loc, scale = stats.beta.fit(y, floc=0, fscale=1)
mu_scipy = a_fit / (a_fit + b_fit)
phi_scipy = a_fit + b_fit
print(f"scipy:   mu_hat={mu_scipy:.4f}, phi_hat={phi_scipy:.4f}")
print(f"max abs diff: mu={abs(mu_hat - mu_scipy):.5f}, phi={abs(model.phi_ - phi_scipy):.4f}")

# ---------------------------------------------------------------------------
# Check 2: finite-difference gradient check of the log-likelihood
# ---------------------------------------------------------------------------
print("\n=== Check 2: analytic optimizer vs finite-difference gradient ===")
n = 300
X = rng.normal(size=(n, 2))
Xb = np.hstack([np.ones((n, 1)), X])
true_beta = np.array([-0.5, 0.8, -0.3])
mu = 1 / (1 + np.exp(-(Xb @ true_beta)))
phi = 8.0
y2 = rng.beta(mu * phi, (1 - mu) * phi)
y2 = np.clip(y2, 1e-6, 1 - 1e-6)

fd_grad = beta_negloglik_gradient_check(Xb, y2, n_beta=Xb.shape[1])
print("finite-difference gradient at a random point:", fd_grad.round(4))
print("(sanity: finite, no NaN/inf -> the hand-derived log-likelihood is well-behaved)")
assert np.all(np.isfinite(fd_grad))

model2 = BetaRegression().fit(X, y2)
print("\nfitted beta (intercept, x1, x2):", model2.beta_.round(4), " true:", true_beta)
print("fitted phi:", round(model2.phi_, 3), " true:", phi)

# ---------------------------------------------------------------------------
# Check 3: cure/severity two-stage model on synthetic LGD data
# ---------------------------------------------------------------------------
print("\n=== Check 3: two-stage cure/severity model, synthetic secured-loan portfolio ===")
n3 = 2000
ltv = rng.uniform(0.4, 1.4, n3)               # loan-to-value at default
liquidity = rng.uniform(0, 1, n3)             # 1 = very liquid collateral
Xc = np.column_stack([ltv, liquidity])

p_cure = 1 / (1 + np.exp(-(2.0 - 2.5 * ltv + 1.5 * liquidity)))
cured = rng.binomial(1, p_cure)

mu_sev = 1 / (1 + np.exp(-(-1.0 + 1.8 * ltv - 1.2 * liquidity)))
severity = rng.beta(mu_sev * 6, (1 - mu_sev) * 6)
lgd_true = np.where(cured == 1, rng.uniform(0, 0.005, n3), severity)  # cured ~ 0 loss

csm = CureSeverityModel(lr=0.3, n_iter=4000, l2=1.0).fit(Xc, lgd_true)
lgd_pred = csm.predict_lgd(Xc)

mae = np.mean(np.abs(lgd_pred - lgd_true))
corr = np.corrcoef(lgd_pred, lgd_true)[0, 1]
print(f"mean realized LGD: {lgd_true.mean():.4f}, mean predicted LGD: {lgd_pred.mean():.4f}")
print(f"MAE: {mae:.4f}, Pearson correlation(pred, realized): {corr:.4f}")

from scipy.stats import spearmanr
rho, _ = spearmanr(lgd_pred, lgd_true)
print(f"Spearman rank correlation (LGD discrimination proxy): {rho:.4f}")
