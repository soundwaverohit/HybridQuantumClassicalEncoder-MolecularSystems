"""This code is a sample pipeline approach we shall be tweaking it further"""

import torch.nn as nn

# Define the classical encoder neural network
class ClassicalEncoder(nn.Module):
    def __init__(self):
        super(ClassicalEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 14),  # First layer with 7 inputs and 14 outputs
            nn.ReLU(),         # Activation function
            nn.Linear(14, 28), # Second layer with 14 inputs and 28 outputs
            nn.ReLU(),         # Activation function
            nn.Linear(28, 56), # Third layer with 28 inputs and 56 outputs
            nn.ReLU(),         # Activation function
            nn.Linear(56, 28), # Fourth layer reducing from 56 to 28 outputs
            nn.ReLU(),         # Activation function
            nn.Linear(28, 14), # Fifth layer reducing from 28 to 14 outputs
            nn.ReLU(),         # Activation function
            nn.Linear(14, 4)   # Final layer reducing to 4 outputs to match the quantum circuit
        )
    
    def forward(self, x):
        return self.fc(x)

encoder = ClassicalEncoder()
#print("The encoder is: ", encoder)
