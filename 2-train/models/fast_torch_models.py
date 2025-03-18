import torch
import torch.nn as nn

# A PyTorch-compatible linear regression model
class FastLinearTorch(nn.Module):
    def __init__(self, input_dim):
        super(FastLinearTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single-layer linear regression

    def forward(self, x):
        return self.linear(x)
