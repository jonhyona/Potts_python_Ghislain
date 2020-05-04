"""
Parameters definition

Different interesting parameters set used must be stored in
'sample_paramters.txt' with explanation of where they come
from and their behavior
"""

# Integration
# Integration
dt = 1.
tSim = 10000

# Network

# Potts units
N = 1000
S = 7
p = 200
# Pattern generation
num_fact = 200
p_fact = 40
dzeta = 0.0000002
a_pf = 0.004
eps = 0.000001
f_russo = 0.1

# Building network
cm = 150
a = 0.25

# Network dynamics
U = 0.1
w = 1.3
# b1 = 0.1
# b2 = 0.005
# b3 = 1
tau_1 = 10
tau_2 = 200
tau_3_A = 5
# tau_1 = 20
# tau_2 = 200
# tau_3_A = 10
tau_3_B = 1e5
g_A = 1.
beta = 11

# Cue
cue_ind = 1
tau = 1
t_0 = 50
g = 10.

random_seed = 2021


def get_parameters():
    return dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, \
        U, T, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, \
        cue_ind, random_seed

def get_f_russo():
    return f_russo
