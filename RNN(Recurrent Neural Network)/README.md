# RNN (Recurrent Neural Network) from First Principles

Companion code for the Medium article "RNN from First Principles: Every Number Solved by Hand (and Where It Breaks)." Every formula is implemented from scratch in plain NumPy and independently verified: brute-force per-timestep recomputation, finite-difference (numerical) gradient checking against the hand-written BPTT backward pass, and a small trained model that actually learns a sequence classification task. The article's central empirical result — the vanishing gradient problem — is measured directly, not just asserted, and sets up the companion LSTM article, which is a genuinely different architecture (not a re-skin of this one) built specifically to fix it.

## Task

Classify a short sequence of monthly credit-utilization ratios (continuing this series' credit-risk theme) as a "distress trend" (rising toward the limit) or "stable" — a 2-class softmax head, same as the CNN article, now driven by a recurrent hidden state instead of a convolution.

## Files

- `rnn_scratch.py` — `rnn_step_forward`/`rnn_step_backward` (one timestep of a tanh RNN cell), `dense_forward`/`dense_backward`, `softmax`/`cross_entropy_loss`/`softmax_cross_entropy_backward`, the `VanillaRNN` class (unrolls the cell over a full sequence, forward and full BPTT backward, and records `||dh_t||` at every timestep), and `numerical_gradient`.
- `data/hand_example.csv` — the 4-month rising-utilization sequence used for the hand-worked example.
- `01_bptt_brute_force_check.py` — verifies `rnn_step_forward` against an independent, unvectorized (pure-Python-loop) implementation.
- `02_hand_example.py` — runs the full forward and BPTT backward pass on the hand example, printing every hidden state, the loss, and every gradient, including the per-timestep gradient norm.
- `03_vanishing_gradient_probe.py` — **the article's key result**: measures `||dh_1||`, the gradient reaching the first timestep, across sequence lengths from 2 to 30. It decays by roughly 5 orders of magnitude, demonstrating the vanishing-gradient problem directly rather than just describing it.
- `04_verify_numerical_gradient.py` — compares every analytic BPTT gradient (`Wxh`, `Whh`, `bh`, `W_out`, `b_out`, and `X`) against `numerical_gradient`'s finite-difference estimate. All match to ~1e-11.
- `05_scratch_demo.py` — trains `VanillaRNN` with plain gradient descent on 200 synthetic utilization sequences; reaches 100% train/test accuracy within 10 epochs.
- `06_library_note.py` — reference-only note (not executed; no PyTorch here) mapping this code onto `nn.RNN`.
- `07_make_figures.py` — the unrolled-network diagram, the hidden-state trajectory, and the vanishing-gradient curve.
- `08_make_table_image.py` — renders the hand-worked example as a table image.

## Why from scratch, and why no PyTorch here

No PyPI access in this sandbox (the same constraint hit for XGBoost, statsmodels, and PyTorch/TensorFlow earlier in this series), so every number is verified three independent ways: brute-force recomputation, finite-difference gradient checking, and a working trained model. `06_library_note.py` documents the exact PyTorch equivalent for reference.

## Run order

```
python3 01_bptt_brute_force_check.py
python3 02_hand_example.py
python3 03_vanishing_gradient_probe.py
python3 04_verify_numerical_gradient.py
python3 05_scratch_demo.py
python3 06_library_note.py
python3 07_make_figures.py
python3 08_make_table_image.py
```

Part of the "from first principles" series: Logistic Regression, Decision Tree, Random Forest, XGBoost, PD, LGD, EAD, CNN, RNN, LSTM.
