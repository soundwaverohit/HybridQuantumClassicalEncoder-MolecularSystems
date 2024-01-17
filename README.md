# HybridQuantumClassicalEncoder-MolecularSystems


## Proposed packages for pipeline: 
- Qiskit for Quantum Circuits
- Pytorch for the Neural Networks


### Sample output of executor.py is:
"""
The encoder is:  ClassicalEncoder(
  (fc): Sequential(
    (0): Linear(in_features=20, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=4, bias=True)
  )
)
The decoder is:  ClassicalDecoder(
  (fc): Sequential(
    (0): Linear(in_features=4, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=20, bias=True)
  )
)
sample data: tensor([0.5105, 0.6529, 0.2404, 0.5552, 0.4041, 0.9692, 0.0205, 0.5498, 0.5644,
        0.8508, 0.5923, 0.5875, 0.7744, 0.0389, 0.8490, 0.4008, 0.8868, 0.6053,
        0.1705, 0.3828])
The encoded parameters are:  tensor([ 0.0317, -0.0045,  0.2034, -0.4625], grad_fn=<ViewBackward0>)
      ┌──────────────┐ ┌─┐         
q_0: ─┤ Rx(0.031671) ├─┤M├─────────
     ┌┴──────────────┴┐└╥┘┌─┐      
q_1: ┤ Rx(-0.0045257) ├─╫─┤M├──────
     └┬─────────────┬─┘ ║ └╥┘┌─┐   
q_2: ─┤ Rx(0.20341) ├───╫──╫─┤M├───
      ├─────────────┴┐  ║  ║ └╥┘┌─┐
q_3: ─┤ Rx(-0.46248) ├──╫──╫──╫─┤M├
      └──────────────┘  ║  ║  ║ └╥┘
c: 4/═══════════════════╩══╩══╩══╩═
                        0  1  2  3 
tensor([0., 0., 0., 0.])
The final output after decoding is: 
tensor([-0.1289,  0.0707, -0.2629, -0.1154, -0.0138,  0.1901, -0.2963, -0.0606,
         0.1109,  0.2114,  0.1229,  0.2007, -0.2612, -0.3134,  0.1790,  0.0500,
        -0.2665, -0.2011,  0.0246, -0.0380], grad_fn=<ViewBackward0>)


"""

