"""
Larger synthetic demo: generate short utilization-ratio sequences, each
either a "distress trend" (rising toward the credit limit, class 1) or
"stable" (flat/noisy around a fixed level, class 0), and train
VanillaRNN with plain gradient descent to see it actually learn --
mirroring the CNN article's training demo, on sequence data instead of
images.
"""
import numpy as np
from rnn_scratch import (rnn_step_forward, rnn_step_backward, dense_forward,
                          dense_backward, softmax, cross_entropy_loss,
                          softmax_cross_entropy_backward)

np.random.seed(42)
T, d, n_h, n_classes = 5, 1, 6, 2


def make_dataset(n):
    X, y = [], []
    for _ in range(n):
        if np.random.rand() < 0.5:
            start = np.random.uniform(0.1, 0.3)
            end = np.random.uniform(0.75, 0.95)
            seq = np.linspace(start, end, T) + np.random.randn(T) * 0.03
            X.append(seq.reshape(T, 1)); y.append(1)
        else:
            level = np.random.uniform(0.2, 0.6)
            seq = level + np.random.randn(T) * 0.05
            X.append(seq.reshape(T, 1)); y.append(0)
    return X, np.array(y)


X_train, y_train = make_dataset(200)
X_test, y_test = make_dataset(60)

Wxh = np.random.randn(n_h, d) * 0.4
Whh = np.random.randn(n_h, n_h) * 0.4
bh = np.zeros(n_h)
W_out = np.random.randn(n_h, n_classes) * 0.4
b_out = np.zeros(n_classes)

lr = 0.1
n_epochs = 80


def forward_backward(X, y_scalar, Wxh, Whh, bh, W_out, b_out):
    y_onehot = np.zeros(n_classes); y_onehot[y_scalar] = 1.0
    Tn = X.shape[0]
    h_prev = np.zeros(n_h)
    hs, a_s = [], []
    for t in range(Tn):
        h_t, a_t = rnn_step_forward(X[t], h_prev, Wxh, Whh, bh)
        hs.append(h_t); a_s.append(a_t); h_prev = h_t
    logits = dense_forward(hs[-1], W_out, b_out)
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, y_onehot)

    dlogits = softmax_cross_entropy_backward(probs, y_onehot)
    dh_last, dW_out, db_out = dense_backward(hs[-1], W_out, dlogits)

    dWxh = np.zeros_like(Wxh); dWhh = np.zeros_like(Whh); dbh = np.zeros_like(bh)
    dh_next = dh_last
    for t in reversed(range(Tn)):
        h_prev_t = hs[t - 1] if t > 0 else np.zeros(n_h)
        dWxh_t, dWhh_t, dbh_t, dx_t, dh_prev = rnn_step_backward(
            dh_next, a_s[t], X[t], h_prev_t, Wxh, Whh)
        dWxh += dWxh_t; dWhh += dWhh_t; dbh += dbh_t
        dh_next = dh_prev

    pred = int(np.argmax(probs))
    return loss, pred, dWxh, dWhh, dbh, dW_out, db_out


print("epoch | mean train loss | train acc | test acc")
for epoch in range(n_epochs):
    total_loss, correct = 0.0, 0
    idx = np.random.permutation(len(X_train))
    for i in idx:
        loss, pred, dWxh, dWhh, dbh, dW_out, db_out = forward_backward(
            X_train[i], y_train[i], Wxh, Whh, bh, W_out, b_out)
        total_loss += loss
        correct += int(pred == y_train[i])
        Wxh -= lr * dWxh
        Whh -= lr * dWhh
        bh -= lr * dbh
        W_out -= lr * dW_out
        b_out -= lr * db_out

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        test_correct = 0
        for i in range(len(X_test)):
            _, pred, *_ = forward_backward(X_test[i], y_test[i], Wxh, Whh, bh, W_out, b_out)
            test_correct += int(pred == y_test[i])
        print(f"{epoch:5d} | {total_loss/len(X_train):.4f}          | "
              f"{correct/len(X_train):.3f}     | {test_correct/len(X_test):.3f}")
