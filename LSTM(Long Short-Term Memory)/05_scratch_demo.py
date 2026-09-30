"""
Larger synthetic demo: same task as the RNN article (classify utilization
sequences as "distress trend" vs "stable"), but on LONGER sequences
(T=15 instead of T=5) where a vanilla RNN's vanishing gradient starts to
bite -- training the LSTM with plain gradient descent to show it still
learns cleanly at a length where the signal would already be badly
attenuated for a vanilla RNN (see 03_gradient_retention_probe.py).
"""
import numpy as np
from lstm_scratch import (lstm_step_forward, lstm_step_backward, dense_forward,
                           dense_backward, softmax, cross_entropy_loss,
                           softmax_cross_entropy_backward)

np.random.seed(42)
T, d, n_h, n_classes = 15, 1, 6, 2


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


def init(shape):
    return np.random.randn(*shape) * 0.3


Wf, Wi, Wg, Wo = [init((n_h, n_h + d)) for _ in range(4)]
bf = np.ones(n_h) * 1.0  # forget-gate bias trick: start biased toward remembering
bi, bg, bo = np.zeros(n_h), np.zeros(n_h), np.zeros(n_h)
W_out = init((n_h, n_classes))
b_out = np.zeros(n_classes)

lr = 0.1
n_epochs = 60


def forward_backward(X, y_scalar, Wf, Wi, Wg, Wo, bf, bi, bg, bo, W_out, b_out):
    y_onehot = np.zeros(n_classes); y_onehot[y_scalar] = 1.0
    Tn = X.shape[0]
    h_prev = np.zeros(n_h); c_prev = np.zeros(n_h)
    hs, caches = [], []
    for t in range(Tn):
        h_t, c_t, cache = lstm_step_forward(X[t], h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo)
        hs.append(h_t); caches.append(cache); h_prev, c_prev = h_t, c_t
    logits = dense_forward(hs[-1], W_out, b_out)
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, y_onehot)

    dlogits = softmax_cross_entropy_backward(probs, y_onehot)
    dh_last, dW_out, db_out = dense_backward(hs[-1], W_out, dlogits)

    dWf = np.zeros_like(Wf); dWi = np.zeros_like(Wi)
    dWg = np.zeros_like(Wg); dWo = np.zeros_like(Wo)
    dbf = np.zeros_like(bf); dbi = np.zeros_like(bi)
    dbg = np.zeros_like(bg); dbo = np.zeros_like(bo)
    dh_next, dc_next = dh_last, np.zeros(n_h)
    for t in reversed(range(Tn)):
        g, dh_prev, dc_prev, dx_t = lstm_step_backward(dh_next, dc_next, caches[t], Wf, Wi, Wg, Wo, n_h)
        dWf += g["dWf"]; dWi += g["dWi"]; dWg += g["dWg"]; dWo += g["dWo"]
        dbf += g["dbf"]; dbi += g["dbi"]; dbg += g["dbg"]; dbo += g["dbo"]
        dh_next, dc_next = dh_prev, dc_prev

    pred = int(np.argmax(probs))
    return loss, pred, dWf, dWi, dWg, dWo, dbf, dbi, dbg, dbo, dW_out, db_out


print(f"Sequence length T={T} (long enough that a vanilla RNN's gradient at t=1")
print("would already be attenuated by >100x -- see 03_gradient_retention_probe.py)\n")
print("epoch | mean train loss | train acc | test acc")
for epoch in range(n_epochs):
    total_loss, correct = 0.0, 0
    idx = np.random.permutation(len(X_train))
    for i in idx:
        (loss, pred, dWf_, dWi_, dWg_, dWo_, dbf_, dbi_, dbg_, dbo_,
         dW_out_, db_out_) = forward_backward(X_train[i], y_train[i], Wf, Wi, Wg, Wo,
                                               bf, bi, bg, bo, W_out, b_out)
        total_loss += loss
        correct += int(pred == y_train[i])
        Wf -= lr * dWf_; Wi -= lr * dWi_; Wg -= lr * dWg_; Wo -= lr * dWo_
        bf -= lr * dbf_; bi -= lr * dbi_; bg -= lr * dbg_; bo -= lr * dbo_
        W_out -= lr * dW_out_; b_out -= lr * db_out_

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        test_correct = 0
        for i in range(len(X_test)):
            _, pred, *_ = forward_backward(X_test[i], y_test[i], Wf, Wi, Wg, Wo, bf, bi, bg, bo, W_out, b_out)
            test_correct += int(pred == y_test[i])
        print(f"{epoch:5d} | {total_loss/len(X_train):.4f}          | "
              f"{correct/len(X_train):.3f}     | {test_correct/len(X_test):.3f}")
