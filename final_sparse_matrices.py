"""Run the model initialized at rest and cued with a pattern
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import numpy.random as rd
# Display time-evolution when integrating
from tqdm import tqdm

# Local modules
from parameters import dt, tSim, N, S, p, t_0, tau, random_seed
import patterns
import initialisation
import iteration

# Required for ssh execution with plots
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

rd.seed(random_seed)

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_uncorrelated()
J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k)

r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, dt_r_i_S_A, \
    dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k = initialisation.network()

print('Intégration')
tS = np.arange(0, tSim, dt)
nT = tS.shape[0]

# Plot parameters
nSnap = nT
r_i_k_plot = np.zeros((nSnap, N*(S+1)))
m_mu_plot = np.zeros((nSnap, p))
theta_i_k_plot = np.zeros((nSnap, N*S))
sig_i_k_plot = np.zeros((nSnap, N*(S+1)))

analyseTime = False
analyseDivergence = False

# Plot parameters
lamb = []
tTrans = []
retrieved_saved = []
max_m_mu_saved = []
max2_m_mu_saved = []
outsider_saved = []
indTrans = 0
cpt_idle = 0
d12 = 0
length = tSim
eta = 0
ind_max_prev = -1

for iT in tqdm(range(nT)):
    iteration.iterate(J_i_j_k_l, delta__ksi_i_mu__k, tS[iT], analyseTime,
                      analyseDivergence, sig_i_k, r_i_k, r_i_S_A,
                      r_i_S_B, theta_i_k, h_i_k, m_mu, dt_r_i_S_A,
                      dt_r_i_S_B, dt_r_i_k_act, dt_theta_i_k)

    r_i_k_plot[iT, :] = r_i_k
    m_mu_plot[iT, :] = m_mu
    theta_i_k_plot[iT, :] = theta_i_k
    sig_i_k_plot[iT, :] = sig_i_k

    if tS[iT] > t_0 + tau:
        ind_max = np.argmax(m_mu)
        max_m_mu = m_mu[ind_max]
        m_mu[ind_max] = - np.inf
        outsider = np.argmax(m_mu)
        max2_m_mu = m_mu[outsider]

        d12 += dt*(max_m_mu - max2_m_mu)

        if ind_max != ind_max_prev:
            tTrans.append(tS[iT])
            lamb.append(max_m_mu)
            retrieved_saved.append(ind_max)
            outsider_saved.append(outsider)
            max_m_mu_saved.append(max_m_mu)
            max2_m_mu_saved.append(max2_m_mu)

            indTrans += 1
            cpt_idle = 0
            eta = 1

        if max_m_mu < .01:
            cpt_idle += 1
            if cpt_idle > nT/10 and nT >= 1000:
                print("latchingDied")
                length = tS[iT]
                break
        else:
            cpt_idle = 0
        ind_max_prev = ind_max

plt.close('all')
units_to_show = np.linspace(0, N-1, 10, dtype=int)
active_states_units_to_show = [i*(S+1) + k for i in units_to_show
                               for k in range(S)]
states_units_to_show = [i*S + k for i in units_to_show
                        for k in range(S)]
plt.figure(1)

ax_overlap = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax_active_input = plt.subplot2grid((3, 2), (1, 0))
ax_thresholds = plt.subplot2grid((3, 2), (1, 1))
ax_active_activity = plt.subplot2grid((3, 2), (2, 0))
ax_inactive_activity = plt.subplot2grid((3, 2), (2, 1))

ax_overlap.plot(tS[:, None], m_mu_plot)
ax_active_input.plot(tS[:, None], r_i_k_plot[:, active_states_units_to_show])
ax_thresholds.plot(tS[:, None], theta_i_k_plot[:, states_units_to_show])
ax_active_activity.plot(tS[:, None], sig_i_k_plot[:,
                                                  active_states_units_to_show])
ax_inactive_activity.plot(tS[:, None], sig_i_k_plot[:, units_to_show + S])

ax_overlap.set_ylabel(r'$m_{\mu}$')
ax_active_input.set_ylabel(r"$r_i^k$")
ax_thresholds.set_ylabel(r"$\theta_i^k$")
ax_active_activity.set_ylabel(r"$\sigma_i^k$")
ax_inactive_activity.set_ylabel("$\sigma_i^0$")

plt.tight_layout()
plt.show()
