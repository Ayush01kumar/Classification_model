import numpy as np


class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using NumPy.

    Model:
        z = Xw + b
        p = sigmoid(z)

    Loss:
        Binary Cross-Entropy

    Optimization:
        Gradient Descent
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        """Numerically stable sigmoid function."""
        z = np.asarray(z)

        result = np.empty_like(z, dtype=float)

        positive = z >= 0
        negative = ~positive

        result[positive] = 1 / (1 + np.exp(-z[positive]))

        exp_z = np.exp(z[negative])
        result[negative] = exp_z / (1 + exp_z)

        return result

    def compute_loss(self, y, z):
        """Numerically stable Binary Cross-Entropy loss."""
        return np.mean(np.logaddexp(0, z) - y * z)

    def fit(self, X, y):
        """Train the model using gradient descent."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):

            # Linear predictor
            z = X @ self.weights + self.bias

            # Predicted probabilities
            probabilities = self.sigmoid(z)

            # Error
            error = probabilities - y

            # Gradients
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            # Parameter update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Store loss
            loss = self.compute_loss(y, z)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """Return probability of class 1."""

        X = np.asarray(X, dtype=float)

        z = X @ self.weights + self.bias

        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Convert probabilities into class predictions."""

        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)


if __name__ == "__main__":

    # Small example
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6]
    ])

    y = np.array([
        0,
        0,
        0,
        1,
        1,
        1
    ])

    model = LogisticRegressionScratch(
        learning_rate=0.01,
        n_iterations=5000
    )

    model.fit(X, y)

    print("Weights:", model.weights)
    print("Bias:", model.bias)

    print("\nProbabilities:")
    print(model.predict_proba(X))

    print("\nPredictions:")
    print(model.predict(X))