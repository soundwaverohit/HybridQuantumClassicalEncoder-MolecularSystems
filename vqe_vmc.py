from qiskit import Aer, execute, QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.opflow import Z, I, StateFn, PauliExpectation, CircuitSampler,Y,X
from qiskit.utils import QuantumInstance
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


def vqe_paramerter_function():

    g = [1.0, -0.2, -0.2, 0.17, 0.18, 0.18]  # Example values, not physically accurate
    hamiltonian_matrix= H_np
    H = MatrixOp(hamiltonian_matrix)

# Constructing the simplified H2 Hamiltonian
    #H = g[0] * (I ^ I) + g[1] * (Z ^ I) + g[2] * (I ^ Z) + g[3] * (Z ^ Z) + g[4] * (Y ^ Y) + g[5] * (X ^ X)

    # Prepare the quantum circuit (ansatz)
    def prepare_ansatz(num_qubits, parameters):
        qc = QuantumCircuit(num_qubits)
        pv = ParameterVector('θ', length=num_qubits * 2)
        idx = 0
        for qubit in range(num_qubits):
            qc.rx(pv[idx], qubit)
            idx += 1
        qc.cz(0, 1)
        for qubit in range(num_qubits):
            qc.rz(pv[idx], qubit)
            idx += 1
        qc = qc.bind_parameters({pv: parameters})
        return qc
    

        # Objective function for minimization
    def objective_function(params):
        ansatz = prepare_ansatz(2, params)
        op = ~StateFn(H) @ StateFn(ansatz)
        expectation = PauliExpectation().convert(op)
        qi = QuantumInstance(Aer.get_backend('statevector_simulator'))
        sampler = CircuitSampler(qi).convert(expectation)
        return sampler.eval().real
    

    # Define the Hamiltonian
    #H = (Z ^ Z) + (Z ^ I) + (I ^ Z)  # Sample model
        # Classical optimization
    initial_params = [0.01] * 4  # Initialize parameters
    result = minimize(objective_function, initial_params, method='COBYLA')

    # Construct the quantum circuit with optimized parameters
    optimized_circuit = prepare_ansatz(2, result.x)

    # Simulate the optimized circuit to get the wavefunction
    backend = Aer.get_backend('statevector_simulator')
    job = execute(optimized_circuit, backend)
    wavefunction = job.result().get_statevector(optimized_circuit)

    return list(result.x), wavefunction


alpha, trial_wavefunctionvqe = vqe_paramerter_function()


def wf_vqe_informed(r1, r2, alpha_vqe):
    '''Computes the trial wavefunction using alpha derived from VQE'''
    norm_r1 = np.linalg.norm(r1)
    norm_r2 = np.linalg.norm(r2)
    r12 = np.linalg.norm(r1 - r2)
    wf = np.exp(-2 * norm_r1) * np.exp(-2 * norm_r2) * np.exp(r12 / (2 * (1 + alpha_vqe * r12)))
    return wf

#define prob density
def prob_density(r1,r2,alpha):
    '''Computes the probability density (not normalized) of the trial wavefunction'''
    return wf_vqe_informed(r1,r2,alpha)**2

#define E local
def E_local(r1,r2,alpha):
    '''Computes the local energy, in terms of r1, r2 and alpha, corresponding to the trial wavefunction'''
    norm_r1 = np.linalg.norm(r1)
    norm_r2 = np.linalg.norm(r2)
    r12 = np.linalg.norm(r1-r2)        
    dot_product = np.dot(r1/norm_r1-r2/norm_r2,r1-r2)
    energy = -4+dot_product/(r12*(1+alpha*r12)**2)-1/(r12*(1+alpha*r12)**3)-1/(4*(1+alpha*r12)**4)+1/r12 
    return energy

def metropolis(N, alpha):
    '''Metropolis algorithm that takes N steps. We start with two random variable within the
    typical length of the problem and then we create a Markov chain taking into account the 
    probability density. At each step we compute the parameters we are interested in.'''
        
    L = 1
    r1 = np.random.rand(3)*2*L-L
    r2 = np.random.rand(3)*2*L-L #random number from -L to L
    E = 0
    E2 = 0
    Eln_average = 0
    ln_average = 0
    rejection_ratio = 0
    step = 0
    max_steps = 500
    
    #Algorithm
    for i in range(N):
        chose = np.random.rand()
        step = step + 1
        if chose < 0.5:
            r1_trial = r1 + 0.5*(np.random.rand(3)*2*L-L)
            r2_trial = r2
        else:
            r2_trial = r2 + 0.5*(np.random.rand(3)*2*L-L)
            r1_trial = r1
        if prob_density(r1_trial,r2_trial,alpha) >= prob_density(r1,r2,alpha):
            r1 = r1_trial
            r2 = r2_trial
        else:
            dummy = np.random.rand()
            if dummy < prob_density(r1_trial,r2_trial,alpha)/prob_density(r1,r2,alpha):
                r1 = r1_trial
                r2 = r2_trial
            else:
                rejection_ratio += 1./N
                
        if step > max_steps:
            E += E_local(r1,r2,alpha)/(N-max_steps)
            E2 += E_local(r1,r2,alpha)**2/(N-max_steps)
            r12 = np.linalg.norm(r1-r2)
            Eln_average += (E_local(r1,r2,alpha)*-r12**2/(2*(1+alpha*r12)**2))/(N-max_steps)
            ln_average += -r12**2/(2*(1+alpha*r12)**2)/(N-max_steps)
    
    return E, E2, Eln_average, ln_average, rejection_ratio

'''Initial parameters'''
#alpha_iterations = 30
alpha_iterations = 6
N_metropolis = 5000
random_walkers = 200
gamma = 0.5

energy_plot = np.array([])
alpha_plot = np.array([])
variance_plot = np.array([])


for i in range(len(alpha)):
    E = 0
    E2 = 0
    dE_dalpha = 0
    Eln = 0
    ln = 0
    rejection_ratio = 0
    
    for j in range(random_walkers): #We use more than one random_walkers in case one gets stuck at some X
        E_met, E2_met, Eln_met, ln_met, rejections_met = metropolis(N_metropolis, alpha[i])
        E += E_met/random_walkers
        E2 += E2_met/random_walkers
        Eln += Eln_met/random_walkers
        ln += ln_met/random_walkers
        rejection_ratio += rejections_met/random_walkers 

    '''Define next alpha'''
    dE_dalpha = 2*(Eln-E*ln)
    print('Alpha: ', alpha[i], '<E>: ', E, 'VarE: ', E2-E**2, 'ratio = ', rejection_ratio)
    #alpha = alpha + 0.05
    #alpha = alpha - gamma*dE_dalpha

    '''Plot'''    
    energy_plot = np.append(energy_plot, E)
    alpha_plot = np.append(alpha_plot, alpha[i])
    variance_plot = np.append(variance_plot, E2-E**2)

    fig1 = plt.figure()

ax1 = fig1.add_subplot(311)
plt.title('Helium atom: evolution of the parameters')
plt.grid()
ax1.plot(alpha_plot, 'g')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Alpha')

ax2 = fig1.add_subplot(312)
plt.grid()
ax2.plot(energy_plot)
ax2.set_xlabel('Timestep')
ax2.set_ylabel('E exp value')
ax2.errorbar(range(len(energy_plot)), energy_plot, yerr=np.sqrt(variance_plot), c='b')

ax3 = fig1.add_subplot(313)
plt.grid()
ax3.plot(variance_plot, 'r')
ax3.set_xlabel('Timestep')
ax3.set_ylabel('Var E')

fig2 = plt.figure()
ax4 = fig2.add_subplot(111)
plt.title('Helium atom: Energy vs. alpha')
plt.grid()
ax4.plot(alpha_plot, energy_plot, 'ro')
ax4.set_xlabel('Alpha')
ax4.set_ylabel('Energy')
