import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import scipy.stats as sts
# Local modules
import file_handling

# plt.ion()
plt.close('all')


def mutual_information(var1_realisations, var2_realisations, p):
    bins1 = np.arange(-0.5, p+.5, 1)
    bins2 = np.arange(-0.5, p+0.5, 1)
    hist1 = np.histogram(var1_realisations, bins=bins1, density=True)[0]
    hist2 = np.histogram(var2_realisations, bins=bins2, density=True)[0]
    cross_hist = np.histogram2d(var1_realisations, var2_realisations,
                                bins=(bins1, bins2), normed=True)[0]
    cross_hist = np.reshape(cross_hist,
                            (cross_hist.shape[0]*cross_hist.shape[1], 1))
    ent_cut = sts.entropy(hist1, base=2)
    ent_shifted = sts.entropy(hist2, base=2)
    ent_joint = ent_cut + ent_shifted - sts.entropy(cross_hist, base=2)
    return ent_cut, ent_shifted, ent_joint

# simulations = ['f1691446ec79cc0cb9d1f3f898c30585',
#                'bd725ecccb665a616415eb80b3742729',
#                '779e267d7fd11b394a96bc18ac9d2261']  # w=1.4

# # simulations = ['6f276611a177a98a02697e035e772a70',
# #                '50a7e2e50bf9b00dff6cd257844d51f7',
# #                '2a123a981c3e2871ff8ff30383ecca93']  #  w=1.3


simulations_above = ['12257f9b2af7fdeaa5ebeec24b71b13c',
                     '2999e6e4eede18f9212d8abdd146e7f4',
                     '779e267d7fd11b394a96bc18ac9d2261']
# Just above the border

simulations_correlated = ['02b6f6e563d04963f75e6967ee04e5f2',
                          '1d02b74d39a60aee6ddb518d63963fc6',
                          '5d4623f054594e3302ebc2323a7c35c8']  # a_pf = 0.4

simulations_correlated = ['86c93c000bbf4b7c36456c248465194d',
                          '7eb9e1a33ed72e9a16c4b856642e4956',
                          '12ec13add87db34c5ad5ca9e64f3e1b3']  # a_pf = 0.2

simulations_correlated = ['bf56183984ac6e3e3e5a0d9978d13e71',
                          '83b3c82f681fa7e87fe6a34f48a804bd',
                          '2805a15bc1193a493a668d717adceff2']  # a_pf = 0.1

simulations_correlated = ['9dca17f2405eb69368bc1eaa88b9d8fc',
                          '3423a65572b2501d87054faedc471399',
                          '557846333a3a3cbd71e0a7e237e02ce5']  # a_pf = 0.05


simulations_correlated = ['b52b35150abd42f0cf538171dad1da5b',
                          'f8d35f84b2626d8311d1cbc33708e2b4',
                          '89c0ad60ba13ea850d51d883aa43007f',
                          '98ac2ddafa16495cce9124b326ba7e30',
                          '3edaf937c6ca25f0a17c0ec12d357e70',
                          '557846333a3a3cbd71e0a7e237e02ce5',
                          '81f17a40c09f63c57166178826580fab',
                          '8c87e68c2af01f21525217a489d76e91',
                          '2805a15bc1193a493a668d717adceff2']

simulations_correlated = ['f2092a45b59f0a93b17179431e64e77e',
                          'daad070458bef11be8c012d8808fd101',
                          '8e3e2d96eae069df663d582d04d7bd62',
                          'ec606c882e32bea57e843d70a860b90d',
                          '121b4158c310037de1aa428457b2f18d',
                          '15d534d88797bf72332ccd745f6220a8',
                          'db1ad0b3dbe738012e19ea6eb8f02bb8',
                          '602f47a63a545b78ec9f5dfe3dfc1bbc',
                          '2233ef41ea4a0d7a0470dac802a03844',
                          '765b4ae39ef5b855d6a41a20057c3041',
                          'cc685ae144f7aa644fc71e9432ce2799',
                          '0d5a5840afacb46019e96820787c1c7b',
                          '72bf257d89270c82c5019e6106de40f6',
                          'af94a9abe9e620848ed1c7f91faafa0c',
                          '5f830645173a7b471271e685f28cac3b']

simulations_correlated = ['f2092a45b59f0a93b17179431e64e77e',
                          'ec606c882e32bea57e843d70a860b90d',
                          'db1ad0b3dbe738012e19ea6eb8f02bb8',
                          '765b4ae39ef5b855d6a41a20057c3041']

simulations_correlated = ['72bf257d89270c82c5019e6106de40f6']


color_s = ['blue', 'orange', 'green']


def plot_overlap(cue, key):
    overlaps = file_handling.load_evolution(cue, key)

    print(overlaps)
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
        file_handling.load_parameters(key)

    tS = np.linspace(0, tSim, nSnap)
    plt.title(r"$a_{pf}$=%.2f, $w$=%.1f, $g_A$=%.1f" % (a_pf, w, g_A))

    plt.plot(tS, overlaps)


n_sim = len(simulations_correlated)
g_A_s = [0.]
apf_s = [0.4]
n_gA = len(g_A_s)
n_apf = len(apf_s)
plt.figure('time_evolution_only_w0')
for ii in range(n_sim):
    key = simulations_correlated[ii]
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
        file_handling.load_parameters(key)
    ind_gA = [i for i in range(len(g_A_s)) if g_A_s[i] == g_A][0]
    ind_apf = [i for i in range(len(apf_s)) if apf_s[i] == a_pf][0]
    # plt.subplot(2, 2, ii+1)
    plot_overlap(0, key)
plt.tight_layout()
plt.show()
