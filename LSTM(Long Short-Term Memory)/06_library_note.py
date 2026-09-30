"""
Reference-only note: how this from-scratch code maps onto PyTorch /
TensorFlow. NOT EXECUTED -- no PyPI access in this sandbox (the
constraint hit throughout this series).

    import torch, torch.nn as nn

    class LSTMTorch(nn.Module):
        def __init__(self, d, n_h, n_classes):
            super().__init__()
            self.lstm = nn.LSTM(d, n_h, batch_first=True)  # like lstm_step_forward, unrolled
            self.fc = nn.Linear(n_h, n_classes)              # like dense_forward

        def forward(self, x):                                 # x: (1, T, d)
            out, (h_T, c_T) = self.lstm(x)
            return self.fc(h_T.squeeze())                      # raw logits

    model = LSTMTorch(d=1, n_h=6, n_classes=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    logits = model(x)
    loss = loss_fn(logits.unsqueeze(0), torch.tensor([1]))
    loss.backward()      # autograd computes exactly the gated BPTT this repo
                          # writes out explicitly in lstm_step_backward
    optimizer.step()

nn.LSTM packs all four gates' weights into two combined matrices
(input-to-hidden and hidden-to-hidden, each 4*n_h wide) purely for
computational efficiency -- mathematically it is exactly the four
separate Wf/Wi/Wg/Wo matrices used here, just concatenated. PyTorch
even applies the SAME forget-gate-bias-of-1 initialization trick used in
05_scratch_demo.py when `nn.LSTM(..., bias=True)`'s defaults are
overridden this way (a common, though not universal, practice), for the
same reason: start the forget gate biased toward "remember everything,"
so gradient has a clear path at the start of training.
"""
print(__doc__)
