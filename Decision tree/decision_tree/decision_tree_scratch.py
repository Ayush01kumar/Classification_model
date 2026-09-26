import numpy as np

# ---------- impurity measures (classification) ----------
def class_probabilities(y, n_classes):
    """p_k = n_k / n"""
    return np.bincount(y, minlength=n_classes) / len(y)

def gini_impurity(y, n_classes):
    """Gini = 1 - sum(p_k^2)"""
    p = class_probabilities(y, n_classes)
    return 1.0 - np.sum(p ** 2)

def entropy(y, n_classes):
    """H = -sum(p_k log2 p_k), with 0*log(0) treated as 0"""
    p = class_probabilities(y, n_classes)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) + 0.0)

# ---------- regression impurity ----------
def variance_impurity(y):
    """mean squared error around the node mean = SSE / n"""
    return np.mean((y - y.mean()) ** 2)

# ---------- split scoring ----------
def weighted_impurity(y_left, y_right, impurity_fn):
    """I_split = (n_L/n) I_L + (n_R/n) I_R"""
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * impurity_fn(y_left) + (len(y_right) / n) * impurity_fn(y_right)

def information_gain(y_parent, y_left, y_right, impurity_fn):
    """Gain = I_parent - I_split"""
    return impurity_fn(y_parent) - weighted_impurity(y_left, y_right, impurity_fn)

def candidate_thresholds(x):
    """midpoints between adjacent DISTINCT sorted values"""
    v = np.unique(x)
    return (v[:-1] + v[1:]) / 2.0

# ---------- tree ----------
class TreeNode:
    def __init__(self, n_samples, impurity, value, proba=None,
                 feature=None, threshold=None, left=None, right=None, gain=0.0):
        self.n_samples, self.impurity, self.value, self.proba = n_samples, impurity, value, proba
        self.feature, self.threshold, self.left, self.right, self.gain = feature, threshold, left, right, gain
    @property
    def is_leaf(self):
        return self.left is None

class DecisionTreeScratch:
    def __init__(self, task="classification", criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_gain=0.0):
        # min_gain=0.0 allows zero-gain splits (as scikit-learn does by default); use a small positive value to forbid them
        self.task, self.criterion = task, criterion
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.min_samples_leaf, self.min_gain = min_samples_leaf, min_gain

    def _impurity_fn(self):
        if self.task == "regression":
            return variance_impurity
        base = gini_impurity if self.criterion == "gini" else entropy
        return lambda y: base(y, self.n_classes_)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        if self.task == "classification":
            self.n_classes_ = int(y.max()) + 1
            y = y.astype(int)
        self.impurity_ = self._impurity_fn()
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def _make_leaf(self, y):
        if self.task == "classification":
            proba = class_probabilities(y, self.n_classes_)
            return TreeNode(len(y), self.impurity_(y), int(np.argmax(proba)), proba)
        return TreeNode(len(y), self.impurity_(y), float(y.mean()))     # leaf = mean(y)

    def _find_best_split(self, X, y):
        best = dict(gain=-np.inf, feature=None, threshold=None)
        parent_impurity = self.impurity_(y)
        for j in range(X.shape[1]):                                    # every feature
            for t in candidate_thresholds(X[:, j]):                    # every candidate threshold
                mask = X[:, j] < t
                y_left, y_right = y[mask], y[~mask]
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                gain = parent_impurity - weighted_impurity(y_left, y_right, self.impurity_)
                if gain > best["gain"]:                                # greedy: keep the best
                    best = dict(gain=gain, feature=j, threshold=t)
        return best

    def _build_tree(self, X, y, depth):
        node = self._make_leaf(y)
        pure = node.impurity == 0.0
        if (pure or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node                                                # stopping condition -> leaf
        best = self._find_best_split(X, y)
        if best["feature"] is None or best["gain"] < self.min_gain - 1e-12:
            return node                                                # no useful split -> leaf
        mask = X[:, best["feature"]] < best["threshold"]
        node.feature, node.threshold, node.gain = best["feature"], best["threshold"], best["gain"]
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def _leaf_for(self, x):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] < node.threshold else node.right
        return node

    def predict_one(self, x):
        return self._leaf_for(x).value

    def predict(self, X):
        return np.array([self.predict_one(x) for x in np.asarray(X, dtype=float)])

    def predict_proba(self, X):
        return np.array([self._leaf_for(x).proba for x in np.asarray(X, dtype=float)])

    def print_tree(self, node=None, indent="", names=None):
        node = node or self.root_
        if node.is_leaf:
            print(f"{indent}leaf: value={node.value}, n={node.n_samples}, impurity={node.impurity:.4f}"); return
        f = names[node.feature] if names else f"x{node.feature}"
        print(f"{indent}{f} < {node.threshold:g}?  (n={node.n_samples}, gain={node.gain:.4f})")
        self.print_tree(node.left, indent + "  yes: ", names); self.print_tree(node.right, indent + "  no:  ", names)
