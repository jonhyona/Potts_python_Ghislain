"""
Parameters definition

Different interesting parameters set used must be stored in
'sample_paramters.txt' with explanation of where they come
from and their behavior
"""
# Integration
dt = 1.                         # Time-step
tSim = 1e5                      # Number of runs
nSnap = min(int(tSim/dt), 2)    # Resampling to plot time-evolution

# Network

# Potts units
N = 1000                        # Number of units in the network
S = 7                           # Number of states
p = 200                         # Number of memorized patterns

# Pattern generation
num_fact = 200                  # Number of factors generated
p_fact = 40                     # Number of children per factor
dzeta = 0.0000002               # Exponential decay of pattern importance
a_pf = 0.004                    # Input sparsity
eps = 0.000001                  # Safety net
f_russo = 0.1                   # Pattern sparsity in Russo2008 algorithm

# Building network
cm = 150                        # Connectivity
a = 0.25                        # Pattern sparsity

# Network dynamics
U = 0.1                         # Threshold
w = 1.4                         # Attractor deepener
tau_1 = 10                      # Activity time-scale
tau_2 = 200                     # Threshold time-scale
tau_3_A = 5                     # Unit threshold fast
tau_3_B = 1e5                   # Unit threshold slow
g_A = 1.                        # Weight between slow-high adapation
beta = 11

# Cue
tau = 10                        # Typical duration of cue
t_0 = 1                         # Time to cue
g = 5.                          # Cue strengh

# Parameters on the cues to use. Useless if run.py is used with the cue
# as a parameter
cue_ind = 1                     # Default pattern to cue
p_0 = 0                         # First cue, useless if cue as a parameter
n_p = 1                         # Number of cues

random_seed = 2021
# The model changed a bit after Russo2008 see h funtion in iteration.py
russo2008_mode = False
