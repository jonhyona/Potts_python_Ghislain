"""Cues the model with all possible patterns and shows the evolution of the
main parameters, as well as statistics on patterns and transitions that occured
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import numpy.random as rd

# Local modules
from parameters import dt, tSim, N, S, p, t_0, tau, random_seed, cm, a, g_A
import patterns
import initialisation
import iteration
import correlations
import seaborn as sns
from tqdm import tqdm

# Required for ssh execution with plots
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

rd.seed(random_seed)

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_uncorrelated()
J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k)

r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
    dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k \
    = initialisation.network(J_i_j_k_l, delta__ksi_i_mu__k)

print('Intégration')
tS = np.arange(0, tSim, dt)
nT = tS.shape[0]

# Plot parameters
nSnap = nT

analyseTime = False
analyseDivergence = False

# Plot parameters
lamb = []                       # Stores crossovers
transition_time = []
retrieved_saved = []
max_m_mu_saved = []
max2_m_mu_saved = []
# Outsider is the pattern with second highest overlap
outsider_saved = []
previously_retrieved_saved = []
transition_counter = 0
cpt_idle = 0
d12 = 0                         # Latching quality metric
length = tSim
eta = 0                         # Did a transition occur?
previously_retrieved = -1

p_0 = 0
n_p = p
for cue_ind in range(p_0, p_0 + n_p):
    print('Cue = pattern ' + str(cue_ind))

    r_i_k_plot = np.zeros((nSnap, N*(S+1)))
    m_mu_plot = np.zeros((nSnap, p))
    theta_i_k_plot = np.zeros((nSnap, N*S))
    sig_i_k_plot = np.zeros((nSnap, N*(S+1)))

    previously_retrieved = cue_ind
    waiting_validation = False
    eta = False
    cpt_idle = 0

    r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
        dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k = initialisation.network(
            J_i_j_k_l, delta__ksi_i_mu__k)

    for iT in tqdm(range(nT)):
        iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT], analyseTime,
                          analyseDivergence, sig_i_k, r_i_k, r_i_S_A,
                          r_i_S_B, theta_i_k, h_i_k, m_mu, dt_r_i_S_A,
                          dt_r_i_S_B, dt_r_i_k_act, dt_theta_i_k, cue_ind, t_0)

        # Saving data for plots
        r_i_k_plot[iT, :] = r_i_k
        m_mu_plot[iT, :] = m_mu
        sig_i_k_plot[iT, :] = sig_i_k
        theta_i_k_plot[iT, :] = theta_i_k

        if tS[iT] > t_0 + 10*tau:
            retrieved_pattern = np.argmax(m_mu)
            max_m_mu = m_mu[retrieved_pattern]
            m_mu[retrieved_pattern] = - np.inf
            outsider = np.argmax(m_mu)
            max2_m_mu = m_mu[outsider]
            d12 += dt*(max_m_mu - max2_m_mu)

            if retrieved_pattern != previously_retrieved \
               and not waiting_validation:
                tmp = [tS[iT], max_m_mu, retrieved_pattern,
                       previously_retrieved, outsider, max_m_mu, max2_m_mu]
                waiting_validation = True
            # Transitions are validated only if the pattern reaches an overlap
            # of 0.5. This avoid to record low-crossover transitions when
            # latching dies
            if waiting_validation and max_m_mu > .5:
                waiting_validation = False
                eta = True
                transition_time.append(tmp[0])
                lamb.append(tmp[1])
                retrieved_saved.append(tmp[2])
                previously_retrieved_saved.append(tmp[3])
                outsider_saved.append(tmp[4])
                max_m_mu_saved.append(tmp[5])
                max2_m_mu_saved.append(tmp[6])

                transition_counter += 1
                cpt_idle = 0
                eta = True
            previously_retrieved = retrieved_pattern

            if max_m_mu < .01:
                cpt_idle += 1
                if cpt_idle > dt*100 and nT >= 1000:
                    print('Latching died')
                    break
            else:
                cpt_idle = 0

s = 2
shift = 1/N/a/5                 # In order categories to be visible in scatter

lamb = np.array(lamb)
bins_lamb = np.linspace(0, 1, 10)
gap = np.logical_and(lamb > 0.25, lamb < 0.55)
C1C2C0 = correlations.cross_correlations(ksi_i_mu)

low_cor = lamb < 0.2
l_low_cor = r'$\lambda < 0.2$'
mid_low_cor = np.logical_and(0.2 <= lamb, lamb < 0.6)
l_mid_low_cor = r'$0.2 \leq \lambda < 0.6$'
mid_high_cor = np.logical_and(0.6 <= lamb, lamb < 0.8)
l_mid_high_cor = r'$0.6 \leq \lambda < 0.8$'
high_cor = 0.8 <= lamb
l_high_cor = r'$0.8 \leq \lambda $'

x0 = np.min(C1C2C0[:, 1]) - 5*shift
x1 = np.max(C1C2C0[:, 1]) + 5*shift
y0 = np.min(C1C2C0[:, 0]) - 5*shift
y1 = np.max(C1C2C0[:, 0]) + 5*shift
bins_y = np.arange(y0-shift, y1 + 1/N/a-shift, 1/N/a)
bins_x = np.arange(x0-shift, x1 + 1/N/a-shift, 1/N/a)
bins_z = np.arange(0, 1, 1/N/a)

XX = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
                                    ksi_i_mu[:, outsider_saved])
YY = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
                                    ksi_i_mu[:, outsider_saved])
ZZ = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
                                  ksi_i_mu[:, outsider_saved])

# %%Plot
plt.close('all')

plt.figure(2)
plt.plot(tS[:, None], m_mu_plot)
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.title(r'$\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
plt.savefig('time_evolution_kwang_il_gA'+str(int(10*g_A))+'.png')

plt.figure(1)
sns.distplot(lamb)
plt.xlabel(r'$\lambda$')
plt.ylabel('Density')
plt.title(r'$\gamma$=' + str(g_A) + ' ; ' + str(len(lamb)) + ' transitions')
plt.savefig('hist_kwang_il_gA'+str(int(10*g_A))+'.png')

# plt.show()
