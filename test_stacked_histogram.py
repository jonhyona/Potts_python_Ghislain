"""Cues the model with all possible patterns and shows the evolution of the
main parameters, as well as statistics on patterns and transitions that occured
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import numpy.random as rd
# Display time-evolution when integrating. Can be deactivated if not possible
# to install package
from tqdm import tqdm

# Local modules
from parameters import dt, tSim, N, S, p, t_0, tau, random_seed, cm, a
import patterns
import initialisation
import iteration
import correlations
import seaborn as sns
import scipy.stats as st
# Required for ssh execution with plots
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

rd.seed(random_seed)

ksi_i_mu, delta__ksi_i_mu__k = patterns.get_uncorrelated()
J_i_j_k_l = initialisation.hebbian_tensor(delta__ksi_i_mu__k)

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
        dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k = initialisation.network(J_i_j_k_l, delta__ksi_i_mu__k)

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

            if retrieved_pattern != previously_retrieved:
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

x0 = np.min(C1C2C0[:, 1])
x1 = np.max(C1C2C0[:, 1])
y0 = np.min(C1C2C0[:, 0])
y1 = np.max(C1C2C0[:, 0])
bins_y = np.arange(y0-shift, y1 + 1/N/a-shift, 1/N/a)
bins_x = np.arange(x0-shift, x1 + 1/N/a-shift, 1/N/a)

XX = correlations.active_diff_state(ksi_i_mu[:, retrieved_saved],
                                    ksi_i_mu[:, outsider_saved])
YY = correlations.active_same_state(ksi_i_mu[:, retrieved_saved],
                                    ksi_i_mu[:, outsider_saved])
ZZ = correlations.active_inactive(ksi_i_mu[:, retrieved_saved],
                                  ksi_i_mu[:, outsider_saved])

# %%Plot
plt.close('all')

plt.figure(1)
plt.subplot(211)
mean_patterns_C1 = np.mean(C1C2C0[:, 0])
mean_patterns_C2 = np.mean(C1C2C0[:, 1])
cat_pattern_1 = np.logical_and(XX < mean_patterns_C2, YY < mean_patterns_C1)
cat_pattern_2 = np.logical_and(XX < mean_patterns_C2, YY >= mean_patterns_C1)
cat_pattern_3 = np.logical_and(XX >= mean_patterns_C2, YY < mean_patterns_C1)
cat_pattern_4 = np.logical_and(XX >= mean_patterns_C2, YY >= mean_patterns_C1)

vals_pattern = [lamb[cat_pattern_1], lamb[cat_pattern_2], lamb[cat_pattern_3],
        lamb[cat_pattern_4]]
n, bins, patches = plt.hist(vals_pattern, 30, stacked=True, density=False)
plt.legend({r'low $C_1$, low $C_2$', 'high $C_1$, low $C_2$',
            'low $C_1$, high $C_2$', 'high $C_1$, high $C_2$'})
plt.ylabel('frequency')
plt.title(r'cm=' + str(cm) + r' $\langle C_1 \rangle_{\mu}$=' + str("%.2f" % mean_patterns_C1) + r' $\langle C_2 \rangle_{\mu}$=' + str("%.2f" % mean_patterns_C2))

plt.subplot(212)
mean_events_C1 = np.mean(YY)
mean_events_C2 = np.mean(XX)
cat_events_1 = np.logical_and(XX < mean_events_C2, YY < mean_events_C1)
cat_events_2 = np.logical_and(XX < mean_events_C2, YY >= mean_events_C1)
cat_events_3 = np.logical_and(XX >= mean_events_C2, YY < mean_events_C1)
cat_events_4 = np.logical_and(XX >= mean_events_C2, YY >= mean_events_C1)

vals_events = [lamb[cat_events_1], lamb[cat_events_2], lamb[cat_events_3], lamb[cat_events_4]]
n, bins, patches = plt.hist(vals_events, 30, stacked=True, density=False)
plt.xlabel(r'\lambda')
plt.ylabel('frequency')
plt.title(r'cm=' + str(cm) + r' $\overline{C_1}$=' + str("%.2f" % mean_events_C1) + r' $\overline{C_2}$=' + str("%.2f" % mean_events_C2))
plt.tight_layout()

plt.figure(2)
plt.subplot(211)
sns.distplot(vals_pattern[0], hist=True, label=r'low $C_1$, low $C_2$', norm_hist=True)
sns.distplot(vals_pattern[1], hist=True, label=r'high $C_1$, low $C_2$', norm_hist=True)
sns.distplot(vals_pattern[2], hist=True, label=r'low $C_1$, high $C_2$', norm_hist=True)
sns.distplot(vals_pattern[3], hist=True, label=r'high $C_1$, high $C_2$', norm_hist=True)
plt.ylabel('frequency')
plt.title(r'cm=' + str(cm) + r' $\langle C_1 \rangle_{\mu}$=' + str("%.2f" % mean_patterns_C1) + r' $\langle C_2 \rangle_{\mu}$=' + str("%.2f" % mean_patterns_C2))
plt.subplot(212)
sns.distplot(vals_events[0], hist=False, label=r'low $C_1$, low $C_2$')
sns.distplot(vals_events[1], hist=False, label=r'high $C_1$, low $C_2$')
sns.distplot(vals_events[2], hist=False, label=r'low $C_1$, high $C_2$')
sns.distplot(vals_events[3], hist=False, label=r'high $C_1$, high $C_2$')
plt.title(r'cm=' + str(cm) + r' $\overline{C_1}$=' + str("%.2f" % mean_events_C1) + r' $\overline{C_2}$=' + str("%.2f" % mean_events_C2))
plt.tight_layout()


plt.figure(3)
plt.subplot(211)
lambda_plot = np.linspace(0, 1, 1000)
estimates_cat_pattern_1 = st.gaussian_kde(lamb[cat_pattern_1])
estimates_cat_pattern_2 = st.gaussian_kde(lamb[cat_pattern_2])
estimates_cat_pattern_3 = st.gaussian_kde(lamb[cat_pattern_3])
estimates_cat_pattern_4 = st.gaussian_kde(lamb[cat_pattern_4])

plt.plot(lambda_plot,
         np.sum(cat_pattern_1) * estimates_cat_pattern_1.evaluate(lambda_plot),
         label=r'low $C_1$, low $C_2$')
plt.plot(lambda_plot,
         np.sum(cat_pattern_2) * estimates_cat_pattern_2.evaluate(lambda_plot),
         label=r'high $C_1$, low $C_2$')
plt.plot(lambda_plot,
         np.sum(cat_pattern_3) * estimates_cat_pattern_3.evaluate(lambda_plot),
         label=r'low $C_1$, high $C_2$')
plt.plot(lambda_plot,
         np.sum(cat_pattern_4) * estimates_cat_pattern_4.evaluate(lambda_plot),
         label=r'high $C_1$, high $C_2$')
plt.plot(lambda_plot, len(lamb) * st.gaussian_kde(lamb).evaluate(lambda_plot), label='Total')

plt.title(r'cm=' + str(cm) + r' $\langle C_1 \rangle_{\mu}$=' + str("%.2f" % mean_patterns_C1) + r' $\langle C_2 \rangle_{\mu}$=' + str("%.2f" % mean_patterns_C2))
plt.legend()

plt.subplot(212)
estimates_cat_events_1 = st.gaussian_kde(lamb[cat_events_1])
estimates_cat_events_2 = st.gaussian_kde(lamb[cat_events_2])
estimates_cat_events_3 = st.gaussian_kde(lamb[cat_events_3])
estimates_cat_events_4 = st.gaussian_kde(lamb[cat_events_4])

plt.plot(lambda_plot,
         np.sum(cat_events_1) * estimates_cat_events_1.evaluate(lambda_plot))
plt.plot(lambda_plot,
         np.sum(cat_events_2) * estimates_cat_events_2.evaluate(lambda_plot))
plt.plot(lambda_plot,
         np.sum(cat_events_3) * estimates_cat_events_3.evaluate(lambda_plot))
plt.plot(lambda_plot,
         np.sum(cat_events_4) * estimates_cat_events_4.evaluate(lambda_plot))
plt.plot(lambda_plot, len(lamb) * st.gaussian_kde(lamb).evaluate(lambda_plot))
plt.title(r'cm=' + str(cm) + r' $\overline{C_1}$=' + str("%.2f" % mean_events_C1) + r' $\overline{C_2}$=' + str("%.2f" % mean_events_C2))
plt.xlabel(r'$\lambda$')
plt.tight_layout()

plt.figure(4)
plt.subplot(211)
y_pattern = []

plt.stackplot(lambda_plot,
              np.sum(cat_pattern_1) * estimates_cat_pattern_1.evaluate(lambda_plot),
              np.sum(cat_pattern_2) * estimates_cat_pattern_2.evaluate(lambda_plot),
              np.sum(cat_pattern_3) * estimates_cat_pattern_3.evaluate(lambda_plot),
              np.sum(cat_pattern_4) * estimates_cat_pattern_4.evaluate(lambda_plot))
plt.legend({r'low $C_1$, low $C_2$', 'high $C_1$, low $C_2$',
            'low $C_1$, high $C_2$', 'high $C_1$, high $C_2$'})

plt.subplot(212)

plt.stackplot(lambda_plot,
              np.sum(cat_events_1) * estimates_cat_events_1.evaluate(lambda_plot),
              np.sum(cat_events_2) * estimates_cat_events_2.evaluate(lambda_plot),
              np.sum(cat_events_3) * estimates_cat_events_3.evaluate(lambda_plot),
              np.sum(cat_events_4) * estimates_cat_events_4.evaluate(lambda_plot))

plt.figure(33)
plt.plot(lambda_plot, np.sum(cat_events_1) *
         estimates_cat_events_1.evaluate(lambda_plot))

plt.show()
