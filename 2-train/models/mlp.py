import torch
import torch.nn as nn

# A PyTorch-compatible simple MLP with one hidden layer
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First linear layer
        self.relu = nn.ReLU()                        # Activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Output linear layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
