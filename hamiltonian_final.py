from executor import decoded_output
from qiskit_nature.second_q.drivers import PySCFDriver
import torch
from pyscf import gto, scf
import numpy as np
import os
import csv
# Define the molecule
mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')
# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Get the Hamiltonian matrix (core Hamiltonian in the AO basis)
H_core = mf.get_hcore()
print('Hamiltonian matrix')
print(H_core)

# Convert Hamiltonian matrix to complex tensor
H_complex = torch.tensor(H_core, dtype=torch.cfloat).real

wavefunction = decoded_output  # Assuming this is a complex tensor
wavefunction_np = wavefunction.detach().numpy()
expectation_value = np.dot(wavefunction_np, np.dot(H_core, wavefunction_np))

# Normalize the wavefunction
norm_wavefunction = wavefunction / torch.sqrt(torch.sum(torch.abs(wavefunction)**2))

# Check if the size of the Hamiltonian matches the size of the wavefunction
# This is crucial, and you need to address this if there's a mismatch
assert H_complex.shape[0] == norm_wavefunction.shape[0], "Size mismatch between Hamiltonian and wavefunction"

# Calculate the energy expectation value
energy = torch.vdot(norm_wavefunction, torch.mv(H_complex, norm_wavefunction)).real
print("Energy found by the hybrid quantum-classical autoencoder: ", expectation_value)


print("True ground state of water: ", )

# Define the molecule
mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')

# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Print the ground state energy
true_energy= mf.e_tot
print("Ground state energy (Hartree-Fock approximation):", mf.e_tot, "Hartree")



import pandas as pd 

df = pd.read_csv("experiment_results.csv")


def name_generator(time):
    string= "experiment_number_"
    string= string+ str(time)
    return string


Experiment_run= expectation_value
# CSV Logging
log_fields = ['Experiment_run', 'true_energy', 'hybrid_model_energy']
log_data = [name_generator(len(df)), true_energy, expectation_value]

# Check if file exists
file_exists = os.path.isfile('experiment_results.csv')

# Write to CSV
with open('experiment_results.csv', 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=log_fields)
    if not file_exists:
        writer.writeheader()
    writer.writerow({field: data for field, data in zip(log_fields, log_data)})