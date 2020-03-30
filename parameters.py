"""
Parameters definition

Different interesting parameters set used must be stored in
'sample_paramters.txt' with explanation of where they come
from and their behavior
"""

dt = 1
tSim = 100000
# objective : tSim = 50000

# Network
# Potts units
N = 200
S = 7
p = 50

# Pattern generation
num_fact = 200
p_fact = 50             # .25*200
dzeta = 0.02
a_pf = 0.25
eps = 0.0000001

# Building network
cm = 25
a = 0.25

# Network dynamics
U = 0.4
T = 0.09
w = 1.8
b1 = 0.1
b2 = 0.005
b3 = 1
tau_1 = 1/b1
tau_2 = 1/b2
tau_3_A = 1/b3
tau_3_B = 1
g_A = 1

beta = 11

# Cue
g = 5.
t_0 = 50
tau = 1
cue_ind = 1

random_seed = 2021

# Carefull : J_ii_kk : not w : has to delete one term!!!

random_seed = 2021


def get_parameters():
    return dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, \
        U, T, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, \
        cue_ind, random_seed
