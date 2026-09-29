"""
NOT EXECUTED in this environment - network access to PyPI was unavailable,
so `statsmodels` (which ships a proper Beta regression, `BetaModel`) could
not be installed here. This file shows what the equivalent production fit
looks like. Run it yourself once statsmodels is installed
(`pip install statsmodels`).

The scratch implementation in lgd_scratch.py (BetaRegression, fit via
scipy.optimize MLE on the reparameterized Beta log-likelihood) was instead
verified two ways: (1) an intercept-only fit matches scipy.stats.beta.fit
exactly (see 04_verify_beta_regression.py, Check 1); (2) a finite-difference
gradient check confirms the hand-derived log-likelihood is implemented
correctly (Check 2).
"""

# import statsmodels.api as sm
# from statsmodels.othermod.betareg import BetaModel
#
# # LGD ~ Beta(mu*phi, (1-mu)*phi), logit link on mu
# X = sm.add_constant(df[["ltv", "liquidity", "seniority"]])
# model = BetaModel(endog=df["lgd"], exog=X, exog_precision=None, link_precision=None)
# result = model.fit()
# print(result.summary())
#
# # equivalent to lgd_scratch.BetaRegression().fit(X_no_const, y).beta_ / .phi_
print(__doc__)
