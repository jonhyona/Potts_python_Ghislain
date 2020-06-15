# coding=utf-8
import numpy as np
import file_handling
import matplotlib.pyplot as plt
import numpy.random as rd
import correlations

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

    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm,
     a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
        file_handling.load_parameters(key)

    coactivation = file_handling.load_coactivation(0, key)
    covariance = file_handling.load_covariance(0, key)

    for ind_cue in range(1, p):
        coactivation += file_handling.load_coactivation(ind_cue, key)
        covariance += file_handling.load_covariance(ind_cue, key)

    coactivation /= p
    covariance /= p

    file_handling.save_mean_coactivation(coactivation, key)
    file_handling.save_mean_covariance(covariance, key)
