"""
Parameters definition

Different interesting parameters set used must be stored in
'sample_paramters.txt' with explanation of where they come
from and their behavior
"""


# Integration
dt = 1
tSim = 100000

# Network
# Potts units
N = 1000
S = 7
p = 200

# Pattern generation
num_fact = 200
p_fact = 200
dzeta = 0.0000002
a_pf = 0.00000001
eps = 0.1

# Building network
cm = 150
a = 0.25

# Network dynamics
U = 0.1
T = 0.09
w = 0.8
tau_1 = 3.33
tau_2 = 100
tau_3_A = 1e6
tau_3_B = 1
g_A = 1

beta = 11

# Cue
g = 5.
t_0 = 50
tau = 1
cue_ind = 1

random_seed = 2021


def get_parameters():
    return dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, \
        U, T, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, \
        cue_ind, random_seed
