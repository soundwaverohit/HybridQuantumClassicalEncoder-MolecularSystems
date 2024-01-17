from executor import decoded_output
from qiskit_nature.second_q.drivers import PySCFDriver
import torch



driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
problem = driver.run()
hamiltonian = problem.hamiltonian
coffs_1= hamiltonian.second_q_op()
dictionary_format = dict(coffs_1)
arrs= []
for elements in dictionary_format:
    arrs.append(dictionary_format[elements])

inputs= torch.tensor(arrs, dtype=torch.float32)

cos = torch.nn.CosineSimilarity(dim=0)
output = cos(inputs, decoded_output)

print("Cosine Similarity:",output)