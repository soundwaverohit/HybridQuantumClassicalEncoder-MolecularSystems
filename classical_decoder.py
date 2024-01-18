import torch
import torch.nn as nn

# Define the classical decoder neural network
class ClassicalDecoder(nn.Module):
    def __init__(self):
        super(ClassicalDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 10),   # First layer with 4 inputs and 10 outputs
            nn.ReLU(),          # Activation function
            nn.Linear(10, 7)   # Second layer with 10 inputs and 20 outputs
        )
    
    def forward(self, x):
        return self.fc(x)

decoder = ClassicalDecoder()
print("The decoder is: ", decoder)
