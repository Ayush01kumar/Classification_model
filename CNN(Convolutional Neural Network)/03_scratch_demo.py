"""
Larger synthetic demo: generate a dataset of small images, each either
containing a vertical edge (class 1) or not (class 0), and train the
TinyCNN with plain gradient descent (no library optimizer) to see the
loss fall and accuracy rise -- an end-to-end sanity check that the
from-scratch forward/backward pass actually learns something, using
only the primitives already verified in 01/02/04.
"""
import numpy as np
from cnn_scratch import (conv2d_forward, conv2d_backward, relu_forward,
                          relu_backward, maxpool_forward, maxpool_backward,
                          dense_forward, dense_backward, softmax,
                          cross_entropy_loss, softmax_cross_entropy_backward)

np.random.seed(42)


def make_dataset(n):
    """6x6 images. Class 1: a vertical edge (left block of 1s, right of 0s,
    edge column randomized) plus noise. Class 0: pure random noise, no
    structured edge."""
    X, y = [], []
    for _ in range(n):
        if np.random.rand() < 0.5:
            edge_col = np.random.choice([2, 3, 4])
            img = np.zeros((6, 6))
            img[:, :edge_col] = 1.0
            img += np.random.randn(6, 6) * 0.15
            X.append(img)
            y.append(1)
        else:
            img = np.random.rand(6, 6) * 0.9  # unstructured, no sharp edge
            X.append(img)
            y.append(0)
    return np.array(X), np.array(y)


X_train, y_train = make_dataset(200)
X_test, y_test = make_dataset(60)

# initialize small random parameters
K = np.random.randn(3, 3) * 0.5
b_conv = 0.0
W_fc = np.random.randn(4, 2) * 0.3
b_fc = np.zeros(2)

lr = 0.05
n_epochs = 60


def forward_backward(X, y_scalar, K, b_conv, W_fc, b_fc):
    y_onehot = np.zeros(2)
    y_onehot[y_scalar] = 1.0

    Z = conv2d_forward(X, K, b_conv)
    A = relu_forward(Z)
    P, argmax = maxpool_forward(A, size=2, stride=2)
    flat = P.flatten()
    logits = dense_forward(flat, W_fc, b_fc)
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, y_onehot)

    dlogits = softmax_cross_entropy_backward(probs, y_onehot)
    dflat, dW_fc, db_fc = dense_backward(flat, W_fc, dlogits)
    dP = dflat.reshape(P.shape)
    dA = maxpool_backward(A.shape, argmax, dP)
    dZ = relu_backward(Z, dA)
    _, dK, db_conv = conv2d_backward(X, K, dZ)

    pred = int(np.argmax(probs))
    return loss, pred, dK, db_conv, dW_fc, db_fc


print("epoch | mean train loss | train acc | test acc")
for epoch in range(n_epochs):
    total_loss, correct = 0.0, 0
    idx = np.random.permutation(len(X_train))
    for i in idx:
        loss, pred, dK, db_conv, dW_fc, db_fc = forward_backward(
            X_train[i], y_train[i], K, b_conv, W_fc, b_fc)
        total_loss += loss
        correct += int(pred == y_train[i])
        K -= lr * dK
        b_conv -= lr * db_conv
        W_fc -= lr * dW_fc
        b_fc -= lr * db_fc

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        test_correct = 0
        for i in range(len(X_test)):
            _, pred, *_ = forward_backward(X_test[i], y_test[i], K, b_conv, W_fc, b_fc)
            test_correct += int(pred == y_test[i])
        print(f"{epoch:5d} | {total_loss/len(X_train):.4f}          | "
              f"{correct/len(X_train):.3f}     | {test_correct/len(X_test):.3f}")

print("\nFinal learned kernel K (should resemble a vertical-edge detector):")
print(np.round(K, 3))
