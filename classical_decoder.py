import torch
import torch.nn as nn

# Define the classical decoder neural network
class ClassicalDecoder(nn.Module):
    def __init__(self):
        super(ClassicalDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 8),    # First layer with 4 inputs and 8 outputs
            nn.ReLU(),          # Activation function
            nn.Linear(8, 16),   # Second layer with 8 inputs and 16 outputs
            nn.ReLU(),          # Activation function
            nn.Linear(16, 32),  # Third layer with 16 inputs and 32 outputs
            nn.ReLU(),          # Activation function
            nn.Linear(32, 16),  # Fourth layer reducing from 32 to 16 outputs
            nn.ReLU(),          # Activation function
            nn.Linear(16, 8),   # Fifth layer reducing from 16 to 8 outputs
            nn.ReLU(),          # Activation function
            nn.Linear(8, 7)     # Final layer reducing to 7 outputs to match the original input size
        )
    
    def forward(self, x):
        return self.fc(x)

decoder = ClassicalDecoder()
#print("The decoder is: ", decoder)
