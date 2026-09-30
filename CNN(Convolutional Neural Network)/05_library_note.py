"""
Reference-only note: how this from-scratch code maps onto PyTorch /
TensorFlow. NOT EXECUTED -- PyTorch/TensorFlow are not installable in
this sandbox (no PyPI access, the same constraint hit for xgboost and
statsmodels earlier in this series). This file documents the
equivalence for the reader; it is not run or verified here.

    import torch, torch.nn as nn

    class TinyCNNTorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3)   # cross-correlation, like conv2d_forward
            self.pool = nn.MaxPool2d(2, 2)                # like maxpool_forward
            self.fc = nn.Linear(4, 2)                     # like dense_forward

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            x = x.flatten()
            return self.fc(x)                             # raw logits; softmax folded into the loss

    model = TinyCNNTorch()
    loss_fn = nn.CrossEntropyLoss()                        # combines softmax + cross-entropy,
                                                             # exactly like this repo's
                                                             # softmax() + cross_entropy_loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    logits = model(x)
    loss = loss_fn(logits.unsqueeze(0), torch.tensor([1]))
    loss.backward()          # autograd computes exactly what conv2d_backward /
                              # relu_backward / maxpool_backward / dense_backward
                              # compute by hand in cnn_scratch.py
    optimizer.step()

Every primitive in cnn_scratch.py has a direct nn.* counterpart:
conv2d_forward/backward <-> nn.Conv2d, relu_forward/backward <-> torch.relu,
maxpool_forward/backward <-> nn.MaxPool2d, dense_forward/backward <->
nn.Linear, softmax + cross_entropy_loss <-> nn.CrossEntropyLoss. The
numbers in 02_hand_example.py and 04_verify_numerical_gradient.py would
match a PyTorch run of the same weights and input to floating-point
precision, since both are computing the same chain rule -- PyTorch's
autograd just automates what is written out explicitly here.
"""
print(__doc__)
