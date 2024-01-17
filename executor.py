"""This is the file to execute the hybrid file"""
import torch

from classical_encoder import encoder
from QuantumCircuit import run_quantum_circuit
from classical_decoder import decoder


# Sample input data
input_data = torch.rand(20)  # Replace with real data
print("sample data:", input_data)

# Encode the input data
encoded_params = encoder(input_data)
print("The encoded parameters are: ", encoded_params)


# Run the quantum circuit
quantum_circuit_output = run_quantum_circuit(encoded_params)
print(quantum_circuit_output)

# Decode the output of the quantum circuit
decoded_output = decoder(quantum_circuit_output)

# Print the final output
print("The final output after decoding is: ")
print(decoded_output)
