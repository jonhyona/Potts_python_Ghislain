import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import seaborn as sns
# Local modules
import file_handling
import networkx as nx
from scipy.optimize import curve_fit
from scipy.special import erf
from statistics import median
import scipy.stats as st

plt.ion()
plt.close('all')
p=200
simulations = ['7ca8d570d1dff01c0133de4031a05b46']
retrieved_saved = [[], [], []]
for kick_seed in range(3):
    retrieved_saved[kick_seed] = \
        file_handling.load_retrieved(kick_seed, simulations[0])

cpt = 0
nmax = 6
for ind_cue in range(p):
    if retrieved_saved[0][ind_cue][0:nmax] == retrieved_saved[1][ind_cue][0:nmax] \
       or retrieved_saved[0][ind_cue][0:nmax] == retrieved_saved[2][ind_cue][0:nmax] \
       or retrieved_saved[1][ind_cue][0:nmax] == retrieved_saved[2][ind_cue][0:nmax]:
        cpt += 1


print(cpt)
print(len(retrieved_saved), len(retrieved_saved[0]),
      len(retrieved_saved[0][0]))
