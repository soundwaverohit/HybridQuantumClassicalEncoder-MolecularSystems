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
    qc = QuantumCircuit(4)
    
    # Apply initial Ry and Rx rotations
    for i in range(4):
        qc.ry(theta[i], i)
        qc.rx(theta[i], i)

    qc.cnot(3,0)
    qc.cnot(1,0)
    qc.cnot(2,1)
    qc.cnot(3,2)

    qc.barrier()

    for i in range(4):
        qc.rx(theta[i], i)
        qc.ry(theta[i], i)
        qc.rx(theta[i], i)

    qc.cnot(3,0)
    qc.cnot(1,0)
    qc.cnot(2,1)
    qc.cnot(3,2)

    qc.barrier()

    for i in range(4):
        qc.rx(theta[i], i)
        qc.ry(theta[i], i)
        qc.rx(theta[i], i)

    qc.cnot(3,0)
    qc.cnot(1,0)
    qc.cnot(2,1)
    qc.cnot(3,2)

    qc.barrier()
    
    

    # Add measurements to all qubits
    qc.measure_all()
    
    # Bind the parameters to the values from the PyTorch model
    param_dict = {theta[i]: params[i].item() for i in range(4)}
    qc_bound = qc.bind_parameters(param_dict)
    
    # Print the quantum circuit
    #print(qc_bound)

    # If you want a visual diagram of the circuit, you can use:
    # circuit_drawer(qc_bound, output='mpl').show()

    # Execute the quantum circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc_bound, backend, shots=1024)
    result = job.result()
    counts = result.get_counts(qc_bound)
    
    # Plot the histogram of results
    plot_histogram(counts)
    
    # Get the most common bitstring
    output_bitstring = max(counts, key=counts.get)
    
    # Convert bitstring to numpy array of integers
    output_data = np.array([int(bit) for bit in output_bitstring])
    
    # Convert to PyTorch tensor
    output_tensor = torch.tensor(output_data, dtype=torch.float32)
    
    return output_tensor


def run_quantum_circuit_and_calculate_expectation_values(params):
    theta = [Parameter(f'θ{i}') for i in range(4)]
    qc = QuantumCircuit(4, 4)

    for i in range(4):
        qc.ry(theta[i], i)
        qc.rx(theta[i], i)

    qc.cnot(3, 0)
    qc.cnot(1, 0)
    qc.cnot(2, 1)
    qc.cnot(3, 2)
    qc.barrier()

    for i in range(4):
        qc.rx(theta[i], i)
        qc.ry(theta[i], i)
        qc.rx(theta[i], i)

    qc.cnot(3,0)
    qc.cnot(1,0)
    qc.cnot(2,1)
    qc.cnot(3,2)

    qc.barrier()

    for i in range(4):
        qc.rx(theta[i], i)
        qc.ry(theta[i], i)
        qc.rx(theta[i], i)

    qc.cnot(3,0)
    qc.cnot(1,0)
    qc.cnot(2,1)
    qc.cnot(3,2)

    qc.barrier()
    
    

    # Add measurements to all qubits
    #qc.measure_all()


    # Do not use qc.measure_all(), as we will be measuring each qubit individually
    for i in range(4):
        qc.measure(i, i)

    param_dict = {theta[i]: params[i].item() for i in range(4)}
    qc_bound = qc.bind_parameters(param_dict)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc_bound, backend, shots=1024)
    result = job.result()
    counts = result.get_counts(qc_bound)

    expectation_values = []
    for i in range(4):
        # Calculate expectation value for each qubit
        p0 = counts.get('0' * i + '0' + '0' * (3 - i), 0) / 1024
        p1 = counts.get('0' * i + '1' + '0' * (3 - i), 0) / 1024
        expectation_value = p0 - p1
        expectation_values.append(expectation_value)

    return torch.tensor(expectation_values, dtype=torch.float32)