"""Bootstrap / out-of-bag probability and the variance of an average of correlated trees.
Run: python 01_bootstrap_oob_variance.py
"""
import numpy as np

print("P(row never drawn) = (1 - 1/n)^n")
for n in (5, 10, 100, 1000):
    print(f"  n={n:5d}: {(1 - 1/n) ** n:.4f}   in-bag share {1 - (1 - 1/n) ** n:.4f}")
print(f"  limit e^-1 = {np.exp(-1):.4f}   in-bag share {1 - np.exp(-1):.4f}")

rng = np.random.default_rng(0)
n = 1000
oob_share = np.mean([1 - len(np.unique(rng.integers(0, n, n))) / n for _ in range(2000)])
print(f"\nSimulation (n={n}, 2000 bootstrap samples): mean OOB share = {oob_share:.4f}")

print("\nVar(average of B trees) = rho*sigma^2 + (1-rho)*sigma^2/B   (sigma^2 = 1)")
for rho in (0, 0.3, 0.6, 1):
    print("  rho=%.1f: " % rho + "  ".join(f"B={B}: {rho + (1 - rho) / B:.3f}" for B in (1, 10, 100)))

# Empirical check with equicorrelated Gaussian 'tree predictions'
rho, B = 0.3, 100
cov = np.full((B, B), rho); np.fill_diagonal(cov, 1.0)
draws = rng.multivariate_normal(np.zeros(B), cov, size=20000)
print(f"\nSimulated variance of the average (rho={rho}, B={B}): {draws.mean(axis=1).var():.3f}  (formula: {rho + (1 - rho) / B:.3f})")
