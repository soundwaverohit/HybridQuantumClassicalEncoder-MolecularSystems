from qiskit_nature.second_q.drivers import PySCFDriver
import torch
from pyscf import gto, scf

mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')
# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

print(mf.get_hcore())
