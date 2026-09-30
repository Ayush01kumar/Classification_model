# CNN (Convolutional Neural Network) from First Principles

Companion code for the Medium article "CNN from First Principles: Every Number Solved by Hand." Every formula in the article is implemented from scratch in plain NumPy and independently verified — by a brute-force quadruple-loop convolution, by finite-difference (numerical) gradient checking against the hand-written backward pass, and by a small trained model that actually learns to detect vertical edges.

## Files

- `cnn_scratch.py` — from-scratch implementation: `conv2d_forward`/`conv2d_backward`, `relu_forward`/`relu_backward`, `maxpool_forward`/`maxpool_backward`, `dense_forward`/`dense_backward`, `softmax`/`cross_entropy_loss`/`softmax_cross_entropy_backward`, the `TinyCNN` end-to-end class, and `numerical_gradient` (a generic central-difference gradient checker).
- `data/hand_example.csv` — the 6x6 vertical-edge toy image used for the hand-worked example.
- `01_conv_brute_force_check.py` — verifies `conv2d_forward` against an independent quadruple-nested-loop implementation and against `scipy.signal.correlate2d`.
- `02_hand_example.py` — runs the full forward and backward pass on the hand example, printing every intermediate value (conv output, ReLU, pooled output, logits, softmax probabilities, loss, and every gradient).
- `03_scratch_demo.py` — generates a 200-image synthetic vertical-edge-detection dataset and trains the `TinyCNN` with plain gradient descent; reaches 100% train/test accuracy and recovers a kernel shaped like a genuine vertical-edge detector.
- `04_verify_numerical_gradient.py` — the core correctness check: compares every analytic gradient (`dK`, `db_conv`, `dW_fc`, `db_fc`, `dX`) against `numerical_gradient`'s finite-difference estimate on a random, non-tied input. All match to ~1e-11. Also documents why the *symmetric* hand-example image produces ties in max pooling that make `dX` (only) mismatch there — an expected non-differentiability, not a bug.
- `05_library_note.py` — reference-only note (not executed; no PyTorch/TensorFlow available in this environment) mapping every scratch function onto its PyTorch `nn.*` equivalent.
- `06_make_figures.py` — generates the pipeline diagram, the hand-example stage heatmaps, the kernel visualization, and the training-loss curve.
- `07_make_table_image.py` — renders the hand-worked example as a table image.

## Why from scratch, and why no PyTorch here

This sandbox has no PyPI access (the same constraint hit earlier in this series for `xgboost` and `statsmodels`), so PyTorch/TensorFlow cannot be installed or executed here. Every number in the article is instead verified three independent ways: brute-force recomputation, finite-difference gradient checking, and a working trained model. `05_library_note.py` documents the exact PyTorch equivalent for reference.

## Run order

```
python3 01_conv_brute_force_check.py
python3 02_hand_example.py
python3 03_scratch_demo.py
python3 04_verify_numerical_gradient.py
python3 05_library_note.py
python3 06_make_figures.py
python3 07_make_table_image.py
```

Part of the "from first principles" series: Logistic Regression, Decision Tree, Random Forest, XGBoost, PD, LGD, EAD, CNN.
