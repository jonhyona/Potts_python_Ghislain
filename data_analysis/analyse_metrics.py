# coding=utf-8
import matplotlib.pyplot as plt
# Standard libraries
import numpy as np
import scipy.stats as sts
# Local modules
import file_handling
import correlations
import seaborn as sns
import pandas as pd

plt.ion()
plt.close('all')
simulations = ['139a320d8afab59bcb18d59268071d94',
               '8d135d1215719ea60f9b7d297f1e9aeb',
               'a63c5077ffe56f59cde4dc33b7a7ac82',
               'ed63915a9b091c1018d50a3d99d48b41',
               '0786d89b52b48b48396d7f7c176fbe55',
               '9e046fe0ec70f37dc196979b42954507',
               'b4babf411ba6416e4fe2c1440a1f9fef',
               '63914c38d1d9c3b1f0f96e97fd35987b',
               '18c9dea2859f57721b6d915c267689f8',
               'f961b5cfb447cb3227dd2b5f5c71c4c1',
               'd158c4222b3dc4d3995869bccdc2f1a3',
               '89c8b67bcd8c40231b7abc89dd30a51e',
               '5e73e76d56736d4451d7b13026e9a22d',
               '72faca25053502fca9ddb571c0bdda2e',
               '7563cf938338f46c3ba9a44c6d891601']

apf_s = np.zeros(len(simulations))
g_A_s = apf_s.copy()
w_s = apf_s.copy()
d12_mean = apf_s.copy()
d12_std = d12_mean.copy()
duration_mean = d12_mean.copy()
duration_std = d12_mean.copy()


def remove_duplicates(mylist):
    return list(dict.fromkeys(mylist))


for ind_key in range(len(simulations)):
    key = simulations[ind_key]
    [d12_s, duration_s] = file_handling.load_metrics(key)
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
        file_handling.load_parameters(key)
    apf_s[ind_key] = a_pf
    d12_s = d12_s/duration_s
    d12_mean[ind_key] = np.mean(d12_s)
    d12_std[ind_key] = np.std(d12_s)
    duration_mean[ind_key] = np.mean(duration_s)
    duration_std[ind_key] = np.std(duration_s)
    g_A_s[ind_key] = g_A
    w_s[ind_key] = w

g_A_values = remove_duplicates(g_A_s)
apf_values = remove_duplicates(apf_s)

plt.figure('Quality')
for g_A in g_A_values:
    ind_s = np.array(g_A_s) == g_A
    w = w_s[ind_s][0]
    plt.plot(apf_s[ind_s], d12_mean[ind_s], label='g_A=%.1f, w=%.1f' % (g_A, w))
    plt.fill_between(apf_s[ind_s], d12_mean[ind_s] + d12_std[ind_s],  d12_mean[ind_s] - d12_std[ind_s], alpha=0.2)

# plt.ylim(0, 3200)
plt.legend()
plt.title(r'Latching quality $d_{12}$')
plt.ylabel(r'$d_{12}$/l')
plt.xlabel(r'$a_{pf}$')

plt.figure('Length')
for g_A in g_A_values:
    ind_s = np.array(g_A_s) == g_A
    w = w_s[ind_s][0]
    plt.plot(apf_s[ind_s], duration_mean[ind_s], label=r'$g_A$=%.1f, w=%.1f' % (g_A,w))
    plt.fill_between(apf_s[ind_s], duration_mean[ind_s] + duration_std[ind_s],  duration_mean[ind_s] - duration_std[ind_s], alpha=0.2)

# plt.ylim(0, 3200)
plt.legend()
plt.title(r'Latching length')
plt.ylabel(r'$l$')
plt.xlabel(r'$a_{pf}$')


plt.figure('C_as')
n_pairs = int(p*(p-1)/2)
n_apf = len(apf_values)
df = np.zeros((len(apf_values)*n_pairs, 2))
for ind_apf in range(len(apf_values)):
    apf = apf_values[ind_apf]
    key = np.array(simulations)[apf_s == apf][0]
    print(apf, key)
    ksi_i_mu, _, _ = file_handling.load_network(key)
    C1C2C0 = correlations.cross_correlations(ksi_i_mu)
    df[ind_apf*n_pairs: (ind_apf+1)*n_pairs, 0] = apf
    df[ind_apf*n_pairs: (ind_apf+1)*n_pairs, 1] = C1C2C0[:, 0]
sns.violinplot(x=df[:, 0], y=df[:, 1])
plt.xlabel(r'$a_{pf}$')
plt.ylabel(r'$C_{as}$')


plt.figure('C_ad')
n_pairs = int(p*(p-1)/2)
n_apf = len(apf_values)
df = np.zeros((len(apf_values)*n_pairs, 2))
for ind_apf in range(len(apf_values)):
    apf = apf_values[ind_apf]
    key = np.array(simulations)[apf_s == apf][0]
    print(apf, key)
    ksi_i_mu, _, _ = file_handling.load_network(key)
    C1C2C0 = correlations.cross_correlations(ksi_i_mu)
    df[ind_apf*n_pairs: (ind_apf+1)*n_pairs, 0] = apf
    df[ind_apf*n_pairs: (ind_apf+1)*n_pairs, 1] = C1C2C0[:, 1]
sns.violinplot(x=df[:, 0], y=df[:, 1])
plt.xlabel(r'$a_{pf}$')
plt.ylabel(r'$C_{ad}$')
# df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

# # Draw Plot
# plt.figure(figsize=(13,10), dpi= 80)
# sns.violinplot(x='class', y='hwy', data=df, scale='width', inner='quartile')

# # Decoration
# plt.title('Violin Plot of Highway Mileage by Vehicle Class', fontsize=22)
# plt.show()
