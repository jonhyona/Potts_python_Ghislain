# coding=utf-8
import numpy as np
import file_handling
import matplotlib.pyplot as plt
import numpy.random as rd
import correlations
from tqdm import tqdm

plt.ion()
plt.close('all')

simulations = ['83058b3d6f4ce563cecce654468e59ec',
               '5fde28fc139c3252f8d52b513c7b2364',
               '6211c3984769aa0bde863c1fa97be8ef',
               '3ae4c5af2e17c42b644210bae0c6c88b',
               'f7fbf477d959473b676fd5e53a243e51',
               '0235947fe67f095afdce360a44932aa7',
               '3f9349f4d58a0590c6575920163dbd45',
               '252c00a8ee9a6dbb553e7166d460b4fe',
               '06381599d71447f5c7c667e3e3106f88',
               'e668a8d56be4b85ea0fe063c0511c617',
               '494a1f3016417558967fc452e83604b0',
               '5c135a6e2604d153a91e4fd757218b49',
               '12a26501a9dd07618c85bd6f324237ed',
               '1d13122682b8d57568e86741055d953b',
               'f61c95aad79795bbe476c2a6692025d5']

# simulations = ['8cda68f5b94f5a6294859908589917b6', 
#                '1818dc5964723deb494a7c19de432430', 
#                'e5ab6ba092eb6742d7f7d09fd5748825', 
#                '3af40b2797084c2ba314e7278b656e54', 
#                '0123cc2bb77249bc63523a0c43d80525', 
#                '97b19f8a5c6ea36b607bc06516cf8a8e', 
#                '6ea3d4a316646e051f7250c5dd72488b', 
#                '9736fe4eff5dd042d99d6ca05197497a', 
#                'b9a4181842479e1234d064800bc31e43', 
#                'ff71c13df64d3f9f005d53374b257de5', 
#                'e90e010be8da4d76e3ef53a4738c7e1d', 
#                '3c917d2304ee6fc333f631fcfffc9b05', 
#                'c3e14af4ed96c8a35fb44ea90efbe996', 
#                'ea86066b7fd721942053f0fee2663dce', 
#                'e9c87e165f69780dfa95fe1c0f4f0fd2']


color_s = ['blue', 'orange', 'green', 'red', 'peru', 'red', 'red', 'red', 'red', 'red']

g_A_s = np.array([0., 0.5, 1.])
apf_s = np.array([0., 0.05, 0.1, 0.2, 0.4])
n_gA = len(g_A_s)
n_apf = len(apf_s)

for ii in range(15):
    print(ii)
    key = simulations[ii]
    [_, durations] = file_handling.load_text(key+'/metrics_cue_%d.txt' % 0)
    ksi_i_mu, delta__ksi_i_mu__k, _, C_i_j = file_handling.load_network(key)


    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
        file_handling.load_parameters(key)

    coactivation = file_handling.load_coactivation_mean(key)
    covariance = file_handling.load_covariance_mean(key)

    covariance_norm = covariance/(coactivation - covariance)
    coactivation_norm = coactivation/(coactivation - covariance)

    corr_proj = file_handling.load_correlation_proj(key)

    parents_children = file_handling.get_parents_children(S, a, a_pf,
                                                          p_fact, N,
                                                          num_fact,
                                                          dzeta)

    shared_parents = np.zeros((p, p), dtype=int)

    for m in range(num_fact):
        for n1 in range(p_fact):
            patt1 = parents_children[m, n1]
            for n2 in range(n1+1, p_fact):
                patt2 = parents_children[m, n2]
                shared_parents[patt1, patt2] += 1
                shared_parents[patt2, patt1] += 1

    shared_density = np.zeros((p*(p-1))//2, dtype=int)
    pair_coactivation = np.zeros((p*(p-1))//2)
    pair_covariance = pair_coactivation.copy()
    pair_coac_norm = pair_coactivation.copy()
    pair_covar_norm = pair_coactivation.copy()
    C1C2C0 = np.zeros(((p*(p-1))//2, 3))
    shared_inact = pair_coactivation.copy()
    cpt = 0

    M = correlations.shared_inactive(ksi_i_mu)

    for patt1 in range(p):
        for patt2 in range(patt1+1, p):
            shared_density[cpt] = shared_parents[patt1, patt2]
            pair_coactivation[cpt] = coactivation[patt1, patt2]
            pair_covariance[cpt] = covariance[patt1, patt2]
            pair_coac_norm[cpt] = coactivation_norm[patt1, patt2]
            pair_covar_norm[cpt] = covariance_norm[patt1, patt2]
            C1C2C0[cpt, 0] = \
                correlations.active_same_state(ksi_i_mu[:, patt1],
                                               ksi_i_mu[:, patt2])
            C1C2C0[cpt, 1] = \
                correlations.active_diff_state(ksi_i_mu[:, patt1],
                                               ksi_i_mu[:, patt2])
            C1C2C0[cpt, 2] = \
                correlations.active_inactive(ksi_i_mu[:, patt1],
                                             ksi_i_mu[:, patt2])
            shared_inact[cpt] = M[patt1, patt2]
            cpt += 1

    ind_gA = [i for i in range(len(g_A_s)) if g_A_s[i] == g_A][0]
    ind_apf = [i for i in range(len(apf_s)) if apf_s[i] == a_pf][0]

    jitterx = 0.15*(rd.rand((p*(p-1)//2))+ind_gA)
    jittery = np.max(coactivation)*(0.05*rd.rand((p*(p-1)//2)))
    # plt.figure('Shared parents')
    # plt.hist(shared_density, bins=np.arange(- 0.5,
    #                                         np.max(shared_density)+0.5,
    #                                         0.25))
    # plt.yscale('log')
    # plt.title('Shared parents')

    plt.figure('Coactivation')
    plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)
    plt.scatter(shared_density+jitterx, pair_coactivation + jittery,
                color=color_s[ind_gA], label=r'$g_A$=%.1f' % g_A, s=1)

    plt.xlabel('Number of shared parents')
    plt.ylabel(r'$<<m^{\mu} m^{\nu}>_t>_p$')
    plt.title(r'$a_{pf}$=%.2f' % a_pf)
    # plt.ylim(1e-3, 1.1)
    # plt.yscale('log')
    # plt.ylim(1e-10, 1e-4)

    plt.figure('Covariance')
    plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)
    plt.scatter(shared_density+jitterx, pair_covariance + jittery,
                color=color_s[ind_gA], label=r'$g_A$=%.1f' % g_A, s=1)


    plt.xlabel('Number of shared parents')
    plt.ylabel(r'$<Cov_t(m^{\mu}, m^{\nu})>_p$')
    plt.title(r'$a_{pf}$=%.2f' % a_pf)
    # plt.yscale('log')
    # plt.ylim(1e-10, 1e-4)

    if g_A == 1.:
        plt.figure('Correlations')
        plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)

        plt.scatter(shared_density+jitterx-0.15*ind_gA, C1C2C0[:,
                                                               0]+jittery,
                    color=color_s[3],
                    label=r'$C_{as}$',
                    s=1)

        plt.scatter(shared_density+0.2+jitterx-0.15*ind_gA, C1C2C0[:,
                                                                   1]+jittery,
                    color=color_s[4],
                    label=r'$C_{ad}$',
                    s=1)
        plt.xlabel('Shared parents')
        plt.ylabel('Correlations')
        plt.title(r'$a_{pf}$=%.2f' % a_pf)

    magic_comb = (1-a/S)**2*C1C2C0[:, 0] - \
        2*a/S*(1-a/S)*(C1C2C0[:, 1] + C1C2C0[:, 2]) \
        + (a/S)**2*shared_inact

    corr_array = np.linspace(np.min(magic_comb), np.max(magic_comb), 15)
    mean_coact_cor = corr_array.copy()
    mean_covar_cor = corr_array.copy()
    std_coact_cor = corr_array.copy()
    std_covar_cor = corr_array.copy()
    for ind_cor in range(len(corr_array)-1):
        indices = np.logical_and(magic_comb >= corr_array[ind_cor],
                                 magic_comb < corr_array[ind_cor + 1])
        mean_coact_cor[ind_cor] = np.mean(pair_coactivation[indices])
        mean_covar_cor[ind_cor] = np.mean(pair_covariance[indices])
        std_coact_cor[ind_cor] = np.std(pair_coactivation[indices])
        std_covar_cor[ind_cor] = np.std(pair_covariance[indices])
    mean_coact_cor[-1] = np.nan
    mean_covar_cor[-1] = np.nan
    std_coact_cor[-1] = np.nan
    std_covar_cor[-1] = np.nan


    # plt.figure('Correlations_coactivations')
    # jitterx = 0.15*np.max(magic_comb)*2*(rd.rand((p*(p-1))//2)-0.5)

    # plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)
    # plt.scatter(magic_comb+jitterx, pair_coac_norm + jittery,
    #             color=color_s[ind_gA], label=r'$g_A$=%.1f'
    #             % g_A, s=1)
    # # plt.ylim(-100, 100)
    # plt.ylabel(r'$<<m^{\mu} m^{\nu}>_t>_p$')
    # plt.xlabel(r'$C_{as}$')

    # plt.figure('Correlation_covariance')
    # plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)
    # # plt.scatter(magic_comb+jitterx, pair_covar_norm + jittery,
    # #             color=color_s[ind_gA], label=r'$g_A$=%.1f'
    # #             % g_A, s=1)
    # # plt.ylim(-100, 100)
    # plt.ylabel(r'$<<m^{\mu} m^{\nu}>_t>_p$')
    # plt.xlabel(r'$C_{as}$')

    plt.figure('Mean attempt coact')
    plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)
    plt.plot(corr_array, mean_coact_cor, color=color_s[ind_gA],
             label=r'$g_A$=%.1f' % g_A)
    plt.fill_between(corr_array, mean_coact_cor + std_coact_cor,
                     mean_coact_cor - std_coact_cor, alpha=0.2,
                     color=color_s[ind_gA])
    plt.title(r'$a_{pf}$=%.2f' % a_pf)

    plt.figure('Mean attempt covar')
    plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)
    plt.plot(corr_array, mean_covar_cor, color=color_s[ind_gA],
             label=r'$g_A$=%.1f' % g_A)
    plt.fill_between(corr_array, mean_covar_cor + std_covar_cor,
                     mean_covar_cor - std_covar_cor, alpha=0.2,
                     color=color_s[ind_gA])
    plt.title(r'$a_{pf}$=%.2f' % a_pf)

plt.figure('Coactivation')
plt.legend()
plt.tight_layout()

plt.figure('Covariance')
plt.legend()
plt.tight_layout()

plt.figure('Correlations')
plt.legend()
plt.tight_layout()

plt.figure('Mean attempt covar')
plt.legend()
plt.tight_layout()

plt.figure('Mean attempt coact')
plt.legend()
plt.tight_layout()
