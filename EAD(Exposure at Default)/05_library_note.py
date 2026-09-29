"""
NOT EXECUTED in this environment - network access to PyPI was unavailable,
so `statsmodels` (BetaModel) could not be installed here, same constraint
as the PD and LGD articles in this series. This file shows the equivalent
production fit for a CCF model. Run it yourself once statsmodels is
installed (`pip install statsmodels`).

The scratch BetaRegression in ead_scratch.py (fit via scipy.optimize MLE
on the reparameterized Beta log-likelihood - identical construction to
the LGD article's severity model) was verified two ways instead: (1) an
intercept-only fit matches scipy.stats.beta.fit exactly; (2) a
finite-difference gradient check confirms the hand-derived log-likelihood.
See 04_verify_beta_regression.py.
"""

# import statsmodels.api as sm
# from statsmodels.othermod.betareg import BetaModel
#
# # CCF ~ Beta(mu*phi, (1-mu)*phi), logit link on mu
# X = sm.add_constant(df[["utilization", "months_to_default", "line_age_years"]])
# model = BetaModel(endog=df["ccf"], exog=X)
# result = model.fit()
# print(result.summary())
print(__doc__)
