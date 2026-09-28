"""
NOT EXECUTED in this environment - network access to PyPI was unavailable,
so packages like `scorecardpy` / `optbinning` could not be installed here.
This file shows what the equivalent production pipeline looks like using
scorecardpy (a widely used open-source credit scoring package). Run it
yourself once the package is installed (`pip install scorecardpy`).

The scratch pipeline in pd_scratch.py implements the same WOE/IV/logistic
/points-scaling/Gini-KS-PSI chain from first principles, and 04_verify_
against_sklearn.py confirms the logistic-regression core matches sklearn
to within numerical tolerance.
"""

# import scorecardpy as sc
#
# # 1. WOE binning (this library uses a chi-merge / tree-based auto-binner)
# bins = sc.woebin(df, y="default")
# sc.woebin_plot(bins)
#
# # 2. Apply WOE transform to train/test
# train_woe = sc.woebin_ply(train, bins)
# test_woe = sc.woebin_ply(test, bins)
#
# # 3. Fit logistic regression on WOE-transformed features
# from sklearn.linear_model import LogisticRegression
# X_train, y_train = train_woe.drop(columns=["default"]), train_woe["default"]
# lr = LogisticRegression(penalty="l2", C=1.0).fit(X_train, y_train)
#
# # 4. Convert to a points scorecard (PDO scaling)
# card = sc.scorecard(bins, lr, X_train.columns, points0=600, odds0=1/50, pdo=20)
# train_score = sc.scorecard_ply(train, card)
#
# # 5. Validate
# perf = sc.perf_eva(y_train, train_score["score"], title="train")  # KS, AUC, Gini plot
# psi_report = sc.perf_psi(train_score, test_score)                 # PSI over time
print(__doc__)
