"""This code is a sample pipeline approach we shall be tweaking it further"""

import torch
import torch.nn as nn

# Define the classical encoder neural network
class ClassicalEncoder(nn.Module):
    def __init__(self):
        super(ClassicalEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 10),  # First layer with 20 inputs and 10 outputs
            nn.ReLU(),          # Activation function
            nn.Linear(10, 4)    # Second layer with 10 inputs and 4 outputs
        )
    
    def forward(self, x):
        return self.fc(x)

encoder = ClassicalEncoder()
