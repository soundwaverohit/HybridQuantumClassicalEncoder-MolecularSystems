# HybridQuantumClassicalEncoder-MolecularSystems


#### TODO
- Implement the HA model from the anamoly detection paper.


## Instructions on running:
- hamiltonian_matrix.py contains the hamiltonian matrix of the water molecule
- Adjust the nueral network parameters and layers in the classical_encoder.py and classical_decoder.py 
- Run the model on learning.py and see the result of the model in the experiment_results.csv file in the last row which containts the true energy, approximated energy by the model and the energy difference.
- The hybrid_autoencoder_pipelinetests.ipynb contains the entire pipeline with a loss function plot as well.

## Repo Information
- The classical encoder, quantum circuit, and decoder are outlined in the files
- The learning.py file contains the training loop to run the hybrid model
- The auto_encoder_notebook will contain the pipeline experimentation before the changes are made to the above files
- The SYK Model hamiltonian is in the SYK hamiltonian folder

## Proposed packages for pipeline: 
- Qiskit for Quantum Circuits
- Pytorch for the Neural Networks


### Sample output of executor.py is:



<img width="741" alt="Screenshot 2024-01-17 at 8 46 15 AM" src="https://github.com/soundwaverohit/HybridQuantumClassicalEncoder-MolecularSystems/assets/30132476/186706eb-7f34-405e-85db-5d944144307a">




Representing the below structure: 
<img width="779" alt="Screenshot 2024-01-17 at 8 46 35 AM" src="https://github.com/soundwaverohit/HybridQuantumClassicalEncoder-MolecularSystems/assets/30132476/a526ed1e-4599-4302-9341-9e75786089dc">
