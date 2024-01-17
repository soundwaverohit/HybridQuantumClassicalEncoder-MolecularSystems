"""This is the file to execute the hybrid file"""
import torch

from classical_encoder import encoder
from QuantumCircuit import run_quantum_circuit
from classical_decoder import decoder


# Sample input data
input_data = torch.rand(20)  # Replace with real data

# Encode the input data
encoded_params = encoder(input_data)

# Run the quantum circuit
quantum_circuit_output = run_quantum_circuit(encoded_params)

# Decode the output of the quantum circuit
decoded_output = decoder(quantum_circuit_output)

# Print the final output
print(decoded_output)
