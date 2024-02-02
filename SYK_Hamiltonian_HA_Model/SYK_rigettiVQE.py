from unicodedata import name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.utils import compute_sample_weight

#from expvalue_on_ansatz import return_lowest_eigenvalue

#Pyquil imports
from pyquil import get_qc, Program
from pyquil.gates import RX, RY, S, T, Z, CNOT, MEASURE, X, Y, RZ, H, CZ
from pyquil.api import WavefunctionSimulator
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator, get_qc
#from pyquil.unitary_tools import lifted_pauli

# Optimizers import 
from scipy.optimize import minimize
import pybobyqa


# Cost function imports
from entropica_qaoa.vqe.cost_function import (PrepareAndMeasureOnWFSim,
                                              PrepareAndMeasureOnQVM)
from entropica_qaoa.qaoa.cost_function import QAOACostFunctionOnWFSim



from pyquil.paulis import PauliTerm, PauliSum, sZ

def hamiltonian_generate(Jx, Jy, Jz, hx, hy, hz):
    pxterm = Jx * (PauliTerm("X",0) * PauliTerm("X",1))
    pyterm = Jy * (PauliTerm("Y",0) * PauliTerm("Y",2))
    pzterm = Jz * (PauliTerm("Z",0) * PauliTerm("Z",3))
    hxterm = hx * (PauliTerm("X",0) + PauliTerm("X",1) + PauliTerm("X",2) + PauliTerm("X",3))
    hyterm = hy * (PauliTerm("Y",0) + PauliTerm("Y",1) + PauliTerm("Y",2) + PauliTerm("Y",3))
    hzterm = hz * (PauliTerm("Z",0) + PauliTerm("Z",1) + PauliTerm("Z",2) + PauliTerm("Z",3))
    pauli_terms=  (-1*(pxterm) - pyterm - pzterm) + hxterm + hyterm + hzterm
    
    return pauli_terms


Jx=0.70710678118
Jy=0.70710678118
Jz=1.0
hx=0.02886751345
hy=0.02886751345
hz=0.02886751345
hamiltonian = hamiltonian_generate(Jx, Jy, Jz, hx, hy, hz)

print(hamiltonian)


program = Program()
depth =4 
params = program.declare("params", memory_type= "REAL", memory_size=depth*6)
#params = ParameterVector('p', depth*6)


for p in range(0, depth*6, 6):
    program += RX(params[p], 0)
    program += RX(params[p], 1)
    program += RX(params[p], 2)
    program += RX(params[p], 3)

    program += H(0)
    program += H(1)

    program += CNOT(0,1)

    program += RZ(params[p+1],1)
        
    program += CNOT(0,1)
        
    program += H(0)
    program += H(1)
        
    program += RY(params[p+2],0)
    program += RY(params[p+2],1)
    program += RY(params[p+2],2)
    program += RY(params[p+2],3)
        
    program += RX(np.pi/2,0)
    program += RX(np.pi/2,2)
        
    program += CNOT(0,2)
    program += RZ(params[p+3],2)
    program += CNOT(0,2)
    
    program += RX(-np.pi/2,0)
    program += RX(-np.pi/2,2)
            
    program += RZ(params[p+4],0)
    program += RZ(params[p+4],1)
    program += RZ(params[p+4],2)
    program += RZ(params[p+4],3)
    
    program += CNOT(0,3)
    program += RZ(params[p+5],3)
    program += CNOT(0,3)

print(program)
    

cost_fun = PrepareAndMeasureOnWFSim(prepare_ansatz=program,
                                    make_memory_map=lambda p: {"params": p},
                                    hamiltonian=hamiltonian,
                                    nshots=8000)


tolerror= 1e-2
true_energy = -1.583135071113909

initial_values = []
for i in range(501):
    initial_values.append(np.random.rand(24))


results = {
    "initial_value":[],
    "solution":[],
    "number_of_evaluations":[]
    
}

i=0
for gamma0 in initial_values:
    soln= pybobyqa.solve(cost_fun, gamma0)
    results["initial_value"].append(gamma0)
    results["solution"].append(soln.f)
    results["number_of_evaluations"].append(soln.nx)
    i=i+1
    print("iteration ",i, " the functional value is: ", soln.f," and number of evaluations is: ", soln.nx)

import pandas as pd

df = pd.DataFrame(results)
print(df)
#df.to_csv('BOBYQA-501values.csv')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

print(find_nearest(df['solution'], true_energy))


import time
start_time = time.time()
gamma0=np.random.rand(24)
soln= pybobyqa.solve(cost_fun, gamma0)
print(soln)
print("--- %s seconds ---" % (time. time() - start_time))