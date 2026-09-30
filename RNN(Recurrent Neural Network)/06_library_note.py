"""
Reference-only note: how this from-scratch code maps onto PyTorch /
TensorFlow. NOT EXECUTED -- no PyPI access in this sandbox (the
constraint hit throughout this series).

    import torch, torch.nn as nn

    class VanillaRNNTorch(nn.Module):
        def __init__(self, d, n_h, n_classes):
            super().__init__()
            self.rnn = nn.RNN(d, n_h, batch_first=True)   # like rnn_step_forward, unrolled
            self.fc = nn.Linear(n_h, n_classes)             # like dense_forward

        def forward(self, x):                                # x: (1, T, d)
            out, h_T = self.rnn(x)                            # h_T: (1, 1, n_h)
            return self.fc(h_T.squeeze())                     # raw logits

    model = VanillaRNNTorch(d=1, n_h=6, n_classes=2)
    loss_fn = nn.CrossEntropyLoss()           # softmax + cross-entropy, like this
                                                # repo's softmax() + cross_entropy_loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    logits = model(x)
    loss = loss_fn(logits.unsqueeze(0), torch.tensor([1]))
    loss.backward()           # autograd performs exactly the BPTT this repo
                               # unrolls by hand in VanillaRNN.backward()
    optimizer.step()

nn.RNN internally does exactly what rnn_step_forward does at every
timestep (tanh(Wxh @ x_t + Whh @ h_prev + bh)), and PyTorch's autograd
computes exactly the chain-rule BPTT this repo writes out explicitly in
rnn_step_backward -- including the same vanishing-gradient behavior for
long sequences, which is a property of the MATH, not of this
implementation; nn.RNN has it too, which is precisely why nn.LSTM (and
this series' separate LSTM article) exists.
"""
print(__doc__)
