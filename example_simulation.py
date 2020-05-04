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
from parameters import dt, tSim, N, S, p, t_0, tau, random_seed, cm, a
import patterns
import initialisation
import iteration
import correlations

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

for cue_ind in range(1):
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

    for iT in range(nT):
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

plt.figure('Lambda distribution')
plt.hist(lamb, bins=20)
plt.xlim((0, 1))
# plt.ylim((0,5))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.title(r'Distribution of $\lambda$, cm = '+str(cm))

plt.figure('Correlations between transition patterns')
ax1 = plt.subplot(221)
ax1.scatter(XX[low_cor]+shift, YY[low_cor]+shift, s=s, c='orange',
            label=l_low_cor)
ax1.scatter(XX[mid_low_cor]+shift, YY[mid_low_cor]-shift, s=s, c='cyan',
            label=l_mid_low_cor)
ax1.scatter(XX[mid_high_cor]-shift, YY[mid_high_cor]+shift, s=s, c='m',
            label=l_mid_high_cor)
ax1.scatter(XX[high_cor], YY[high_cor], s=s, c='g', label=l_high_cor)
ax1.legend()

ax1.set_ylabel('C1')
ax1.set_xlabel('C2')
ax1.set_xlim(x0, x1)
ax1.set_ylim(y0, y1)
ax1.set_title('Correlations between transition patterns')

ax2 = plt.subplot(222)
plt.hist2d(XX, YY, bins=(bins_x, bins_y))
ax2.set_xlabel('C2')
ax2.set_xlim(x0, x1)
ax2.set_ylim(y0, y1)
plt.colorbar()

ax3 = plt.subplot(223)
plt.scatter(C1C2C0[:, 1], C1C2C0[:, 0], s=s)
ax3.set_xlim(x0, x1)
ax3.set_ylim(y0, y1)
ax3.set_title('Correlations between all patterns')

ax4 = plt.subplot(224)
plt.hist2d(C1C2C0[:, 1], C1C2C0[:, 0], bins=(bins_x, bins_y))
plt.colorbar()
ax4.set_xlim(x0, x1)
ax4.set_ylim(y0, y1)

plt.figure('Lambda, C')
plt.subplot(131)
plt.hist2d(YY, lamb, bins=(bins_y, bins_lamb))
plt.xlabel('C1')
plt.ylabel(r'$\lambda$')
plt.colorbar()
plt.subplot(132)
plt.hist2d(XX, lamb, bins=(bins_x, bins_lamb))
plt.xlabel('C2')
plt.colorbar()
plt.subplot(133)
plt.hist2d(ZZ, lamb, bins=(bins_z, bins_lamb))
plt.xlabel('Active Inactive')
plt.colorbar()

plt.tight_layout()

plt.figure('Compared distributions')
max_C1 = np.max(C1C2C0[:, 0])
max_C2 = np.max(C1C2C0[:, 1])
max_C0 = np.max(C1C2C0[:, 2])

min_C1 = np.min(C1C2C0[:, 0])
min_C2 = np.min(C1C2C0[:, 1])
min_C0 = np.min(C1C2C0[:, 2])

ax_C1_pattern = plt.subplot(231)
ax_C2_pattern = plt.subplot(232)
ax_C0_pattern = plt.subplot(233)

ax_C1_event = plt.subplot(234)
ax_C2_event = plt.subplot(235)
ax_C0_event = plt.subplot(236)

ax_C1_pattern.hist(C1C2C0[:, 0])
ax_C2_pattern.hist(C1C2C0[:, 1])
ax_C0_pattern.hist(C1C2C0[:, 2])

ax_C1_event.hist(YY)
ax_C2_event.hist(XX)
ax_C0_event.hist(ZZ)

ax_C1_event.set_xlim(min_C1, max_C1)
ax_C2_event.set_xlim(min_C1, max_C2)
ax_C0_event.set_xlim(min_C0, max_C0)

ax_C1_pattern.set_xlim(min_C1, max_C1)
ax_C2_pattern.set_xlim(min_C1, max_C2)
ax_C0_pattern.set_xlim(min_C0, max_C0)

plt.tight_layout()

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
ax_inactive_activity.plot(tS[:, None], r_i_k_plot[:, (S+1)*units_to_show + S])

ax_overlap.set_ylabel(r'$m_{\mu}$')
ax_overlap.set_title(
    'Time evolution of the network parameters, with last pattern cued')
ax_active_input.set_ylabel(r"$r_i^k$")
ax_thresholds.set_ylabel(r"$\theta_i^k$")
ax_active_activity.set_ylabel(r"$\sigma_i^k$")
ax_inactive_activity.set_ylabel(r"$\theta_i^0$")

plt.tight_layout()
plt.show()


plt.figure('histogram')
plt.hist2d(C1C2C0[:, 0], C1C2C0[:, 1])
plt.xlabel('C1')
plt.ylabel('C2')
plt.xlim(0, 0.4)
plt.ylim(0, 0.4)
plt.colorbar()
plt.show()
