import numpy as np
from decision_tree_scratch import DecisionTreeScratch, gini_impurity, entropy


class _RandomizedTree(DecisionTreeScratch):
    """Decision tree that searches only a random subset of features at EVERY split."""
    def __init__(self, max_features=None, rng=None, **kw):
        super().__init__(**kw)
        self.max_features, self.rng = max_features, rng

    def _features_to_try(self, n_features):
        m = n_features if self.max_features is None else self.max_features
        return self.rng.choice(n_features, size=m, replace=False)


class RandomForestScratch:
    """Random forest = bagging (bootstrap + average) + random feature subsets at each split."""

    def __init__(self, n_estimators=100, max_features="sqrt", criterion="gini", max_depth=None,
                 min_samples_leaf=1, bootstrap=True, random_state=0):
        self.n_estimators, self.max_features, self.criterion = n_estimators, max_features, criterion
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf
        self.bootstrap, self.random_state = bootstrap, random_state

    def _m(self, p):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(p)))                 # m = floor(sqrt(p))
        if self.max_features is None:
            return p
        return int(self.max_features)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(int)
        n, p = X.shape
        self.n_features_ = p
        self.n_classes_ = int(y.max()) + 1
        rng = np.random.default_rng(self.random_state)
        self.trees_, self.oob_masks_ = [], []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)     # bootstrap sample
            oob = np.ones(n, bool); oob[idx] = False                            # rows left out (~36.8%)
            tree = _RandomizedTree(max_features=self._m(p), rng=rng, criterion=self.criterion,
                                   max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree); self.oob_masks_.append(oob)
        self._oob_score(X, y)
        return self

    def _tree_proba(self, tree, X):
        pr = tree.predict_proba(X)                                              # pad if a class was absent
        out = np.zeros((len(X), self.n_classes_)); out[:, :pr.shape[1]] = pr
        return out

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return np.mean([self._tree_proba(t, X) for t in self.trees_], axis=0)   # average of tree probabilities

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def _oob_score(self, X, y):
        n = len(y); acc = np.zeros((n, self.n_classes_)); cnt = np.zeros(n)
        for tree, oob in zip(self.trees_, self.oob_masks_):
            if oob.any():
                acc[oob] += self._tree_proba(tree, X[oob]); cnt[oob] += 1
        seen = cnt > 0
        self.oob_decision_ = acc[seen] / cnt[seen, None]
        self.oob_score_ = float(np.mean(self.oob_decision_.argmax(axis=1) == y[seen]))   # OOB accuracy

    @property
    def feature_importances_(self):
        """Mean decrease in impurity: sum over nodes of n_node * gain, normalised per tree, averaged."""
        imps = []
        for t in self.trees_:
            imp = np.zeros(self.n_features_)
            self._accumulate(t.root_, imp)
            s = imp.sum()
            imps.append(imp / s if s > 0 else imp)
        return np.mean(imps, axis=0)

    def _accumulate(self, node, imp):
        if node.is_leaf:
            return
        imp[node.feature] += node.n_samples * node.gain
        self._accumulate(node.left, imp); self._accumulate(node.right, imp)
