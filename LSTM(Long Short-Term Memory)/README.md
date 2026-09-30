# LSTM (Long Short-Term Memory) from First Principles

Companion code for the Medium article "LSTM from First Principles: How Gates Fix What RNNs Break." Every gate equation is implemented from scratch in plain NumPy and independently verified: brute-force per-timestep recomputation, finite-difference (numerical) gradient checking against the hand-written BPTT backward pass, and a small trained model that learns on sequences long enough to have broken the companion RNN article's vanilla architecture. This repo also directly measures — not just claims — that the LSTM retains far more gradient signal than a vanilla RNN at the same sequence lengths.

## Task

Same as the RNN article, for direct comparison: classify a short sequence of monthly credit-utilization ratios as a "distress trend" (rising toward the limit) or "stable", via a 2-class softmax head.

## Files

- `lstm_scratch.py` — `lstm_step_forward`/`lstm_step_backward` (one timestep: forget/input/output gates, candidate cell content, and the additive cell-state update), `dense_forward`/`dense_backward`, `softmax`/`cross_entropy_loss`/`softmax_cross_entropy_backward`, the `LSTM` class (unrolls over a sequence, forward and full BPTT backward, recording `||dh_t||` and `||dc_t||` at every timestep), and `numerical_gradient`.
- `data/hand_example.csv` — the same 4-month rising-utilization sequence used in the RNN article, so the two architectures can be compared on identical input.
- `01_gate_brute_force_check.py` — verifies `lstm_step_forward` (all four gates, cell state, hidden state) against an independent, unvectorized (pure-Python-loop) implementation.
- `02_hand_example.py` — runs the full forward and BPTT backward pass on the hand example, printing every gate, the cell state, the hidden state, the loss, and every gradient.
- `03_gradient_retention_probe.py` — **the article's key result**: reruns the RNN article's vanishing-gradient experiment for both a vanilla RNN and this LSTM on the same weight scale and sequence lengths. At T=30 the LSTM retains roughly 1,270x more gradient at the first timestep than the RNN.
- `04_verify_numerical_gradient.py` — compares every analytic BPTT gradient (`Wf`, `Wi`, `Wg`, `Wo`, all four biases, `W_out`, `b_out`, and `X`) against `numerical_gradient`'s finite-difference estimate. All 11 match to ~1e-11.
- `05_scratch_demo.py` — trains the LSTM with plain gradient descent on 200 synthetic sequences of length T=15 (long enough that a vanilla RNN's gradient at the first timestep would already be attenuated by >100x); reaches 100% train/test accuracy within 10 epochs regardless.
- `06_library_note.py` — reference-only note (not executed; no PyTorch here) mapping this code onto `nn.LSTM`.
- `07_make_figures.py` — the gate/cell-state diagram, the gate-and-state trajectory on the hand example, and the RNN-vs-LSTM gradient-retention comparison.
- `08_make_table_image.py` — renders the hand-worked example and the retention comparison as a table image.

## Why from scratch, and why no PyTorch here

No PyPI access in this sandbox (the same constraint hit for XGBoost, statsmodels, and PyTorch/TensorFlow earlier in this series), so every number is verified three independent ways: brute-force recomputation, finite-difference gradient checking, and a working trained model. `06_library_note.py` documents the exact PyTorch equivalent for reference.

## Run order

```
python3 01_gate_brute_force_check.py
python3 02_hand_example.py
python3 03_gradient_retention_probe.py
python3 04_verify_numerical_gradient.py
python3 05_scratch_demo.py
python3 06_library_note.py
python3 07_make_figures.py
python3 08_make_table_image.py
```

Part of the "from first principles" series: Logistic Regression, Decision Tree, Random Forest, XGBoost, PD, LGD, EAD, CNN, RNN, LSTM.
