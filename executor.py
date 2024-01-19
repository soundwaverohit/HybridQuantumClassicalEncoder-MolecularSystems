"""This is the file to execute the hybrid file"""
import torch

from classical_encoder import encoder
from QuantumCircuit import run_quantum_circuit
from classical_decoder import decoder
from qiskit_nature.second_q.drivers import PySCFDriver

#from hamiltonian_initial import inputs



"""
# Sample input data
input_data = torch.rand(20)  # Replace with real data
print("sample data:", input_data)
"""

inputs= torch.tensor([ 0.3679, -0.0602,  0.6200,  0.1083, -0.0054,  0.0107,  0.1241])
print("input coefficient data from the hamiltonian:", inputs)



# Encode the input data
encoded_params = encoder(inputs)
print("The encoded parameters are: ", encoded_params)


# Run the quantum circuit
quantum_circuit_output = run_quantum_circuit(encoded_params)
print(quantum_circuit_output)

# Decode the output of the quantum circuit
decoded_output = decoder(quantum_circuit_output)

# Print the final output
print("The final output after decoding is: ")
print(decoded_output)
