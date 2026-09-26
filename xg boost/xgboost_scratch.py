"""XGBoost from first principles (exact greedy algorithm, second-order Taylor objective).

Objective per round t:   L = sum_i [ g_i f(x_i) + 0.5 h_i f(x_i)^2 ] + gamma*T + 0.5*lambda*sum_j w_j^2
Optimal leaf weight:     w* = -G / (H + lambda)
Split gain:              0.5 * [ G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - (G_L+G_R)^2/(H_L+H_R+lam) ] - gamma
Prediction:              F_t(x) = F_{t-1}(x) + eta * f_t(x)      (margin; sigmoid for classification)

Supported objectives: "binary:logistic" and "reg:squarederror".
General XGBoost theory is in the module; library-specific defaults are noted in comments.
Not implemented (on purpose): histogram/approximate splits, sparsity-aware default directions,
post-pruning, multiclass, L1 (alpha), column sampling per level/node.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "weight", "gain", "cover", "G", "H")

    def __init__(self):
        self.feature = None; self.threshold = None; self.left = None; self.right = None
        self.weight = 0.0; self.gain = 0.0; self.cover = 0.0; self.G = 0.0; self.H = 0.0


class BoostedTree:
    """One regression tree fitted to (g, h) with the XGBoost structure score."""

    def __init__(self, max_depth=6, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
        self.max_depth, self.lam, self.gamma, self.mcw = max_depth, reg_lambda, gamma, min_child_weight

    def fit(self, X, g, h, feature_ids=None):
        self.n_features_ = X.shape[1]
        self.feature_ids_ = np.arange(X.shape[1]) if feature_ids is None else np.asarray(feature_ids)
        self.root_ = self._build(X, g, h, np.arange(len(g)), 0)
        return self

    def _build(self, X, g, h, idx, depth):
        node = _Node()
        node.G, node.H, node.cover = g[idx].sum(), h[idx].sum(), h[idx].sum()
        node.weight = -node.G / (node.H + self.lam)                      # optimal leaf weight
        if depth >= self.max_depth or len(idx) < 2:
            return node
        best = self._best_split(X, g, h, idx, node.G, node.H)
        if best is None:
            return node
        gain, j, t = best
        mask = X[idx, j] < t
        node.feature, node.threshold, node.gain = j, t, gain
        node.left = self._build(X, g, h, idx[mask], depth + 1)
        node.right = self._build(X, g, h, idx[~mask], depth + 1)
        return node

    def _best_split(self, X, g, h, idx, G, H):
        lam, parent = self.lam, G * G / (H + self.lam)
        best = None
        for j in self.feature_ids_:
            order = idx[np.argsort(X[idx, j], kind="mergesort")]
            xs = X[order, j]
            GL = np.cumsum(g[order])[:-1]; HL = np.cumsum(h[order])[:-1]
            GR, HR = G - GL, H - HL
            valid = (xs[1:] > xs[:-1]) & (HL >= self.mcw) & (HR >= self.mcw)     # min_child_weight on Hessian sum
            if not valid.any():
                continue
            gain = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - parent) - self.gamma
            gain = np.where(valid, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > 1e-12 and (best is None or gain[k] > best[0] + 1e-12):
                best = (float(gain[k]), int(j), float((xs[k] + xs[k + 1]) / 2))    # midpoint threshold, x < t goes left
        return best

    def predict(self, X):
        out = np.empty(len(X))
        for i, row in enumerate(X):
            n = self.root_
            while n.left is not None:
                n = n.left if row[n.feature] < n.threshold else n.right
            out[i] = n.weight
        return out

    def leaves(self):
        stack, out = [self.root_], []
        while stack:
            n = stack.pop()
            if n.left is None: out.append(n)
            else: stack += [n.left, n.right]
        return out

    def splits(self):
        stack, out = [self.root_], []
        while stack:
            n = stack.pop()
            if n.left is not None: out.append(n); stack += [n.left, n.right]
        return out


class XGBoostScratch:
    def __init__(self, objective="binary:logistic", n_estimators=100, learning_rate=0.3, max_depth=6,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0, base_score=None,
                 subsample=1.0, colsample_bytree=1.0, random_state=0):
        self.objective, self.n_estimators, self.eta = objective, n_estimators, learning_rate
        self.max_depth, self.reg_lambda, self.gamma, self.mcw = max_depth, reg_lambda, gamma, min_child_weight
        self.base_score, self.subsample, self.colsample = base_score, subsample, colsample_bytree
        self.random_state = random_state

    # ---- objective-specific pieces -------------------------------------------------------------
    def _grad_hess(self, y, F):
        if self.objective == "binary:logistic":
            p = sigmoid(F)
            return p - y, p * (1 - p)                       # g = p - y,  h = p(1-p)
        return F - y, np.ones_like(F)                       # squared error: g = F - y, h = 1

    def _loss(self, y, F):
        if self.objective == "binary:logistic":
            p = np.clip(sigmoid(F), 1e-15, 1 - 1e-15)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return float(np.mean((y - F) ** 2))

    def _init_margin(self, y):
        if self.base_score is None:
            return 0.0 if self.objective == "binary:logistic" else float(np.mean(y))   # p=0.5 -> margin 0
        b = float(self.base_score)
        return float(np.log(b / (1 - b))) if self.objective == "binary:logistic" else b

    # ---- fit / predict -------------------------------------------------------------------------
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        self.n_features_ = p
        self.f0_ = self._init_margin(y)
        F = np.full(n, self.f0_)
        self.trees_, self.train_loss_ = [], [self._loss(y, F)]
        for _ in range(self.n_estimators):
            g, h = self._grad_hess(y, F)
            rows = np.arange(n) if self.subsample >= 1 else np.sort(rng.choice(n, max(1, int(self.subsample * n)), replace=False))
            cols = None if self.colsample >= 1 else np.sort(rng.choice(p, max(1, int(round(self.colsample * p))), replace=False))
            tree = BoostedTree(self.max_depth, self.reg_lambda, self.gamma, self.mcw)
            tree.fit(X[rows], g[rows], h[rows], cols)
            # rebuild on full X would change splits; leaf weights already from the sampled rows (same as XGBoost's subsample)
            F = F + self.eta * tree.predict(X)
            self.trees_.append(tree); self.train_loss_.append(self._loss(y, F))
        return self

    def decision_function(self, X, n_trees=None):
        X = np.asarray(X, float)
        F = np.full(len(X), self.f0_)
        for t in self.trees_[:n_trees]:
            F = F + self.eta * t.predict(X)
        return F

    def predict_proba(self, X, n_trees=None):
        p = sigmoid(self.decision_function(X, n_trees))
        return np.column_stack([1 - p, p])

    def predict(self, X, n_trees=None):
        if self.objective == "binary:logistic":
            return (self.predict_proba(X, n_trees)[:, 1] >= 0.5).astype(int)
        return self.decision_function(X, n_trees)

    # ---- importance ----------------------------------------------------------------------------
    def feature_importances_(self, kind="gain"):
        """Total gain per feature (kind='gain'), split count ('weight') or summed Hessian cover ('cover')."""
        imp = np.zeros(self.n_features_)
        for t in self.trees_:
            for n in t.splits():
                imp[n.feature] += n.gain if kind == "gain" else (1 if kind == "weight" else n.cover)
        s = imp.sum()
        return imp / s if s > 0 else imp
