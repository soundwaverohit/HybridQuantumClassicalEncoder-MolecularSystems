from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter
import numpy as np
import torch 

# Define a function to execute the quantum circuit
def run_quantum_circuit(params):
    # Create a list of parameters for the quantum circuit
    theta = [Parameter(f'θ{i}') for i in range(4)]
    
    # Create a quantum circuit with 4 qubits and 4 classical bits
    qc = QuantumCircuit(4, 4)
    
    # Add parameterized rotations and a measurement to each qubit
    for i in range(4):
        qc.rx(theta[i], i)
        qc.measure(i, i)
    
    # Bind the parameters and execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    t_params = {theta[i]: params[i].item() for i in range(4)}
    circuit = qc.bind_parameters(t_params)
    
    # Execute the quantum circuit
    job = execute(circuit, backend, shots=1024)
    result = job.result()
    counts = result.get_counts(circuit)
    
    # Plot the histogram of results
    plot_histogram(counts)
    
    # Get the most common bitstring
    output_bitstring = max(counts, key=counts.get)
    
    # Convert bitstring to numpy array of integers
    output_data = np.array([int(bit) for bit in output_bitstring])
    
    # Convert to PyTorch tensor
    output_tensor = torch.tensor(output_data, dtype=torch.float32)
    
    return output_tensor
