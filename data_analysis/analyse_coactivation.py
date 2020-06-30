# coding=utf-8
import numpy as np
import file_handling
import matplotlib.pyplot as plt
import numpy.random as rd
import iteration
from scipy.spatial import ConvexHull, convex_hull_plot_2d

plt.ion()
plt.close('all')

# simulations = ['83058b3d6f4ce563cecce654468e59ec',
#                '5fde28fc139c3252f8d52b513c7b2364',
#                '6211c3984769aa0bde863c1fa97be8ef',
#                '3ae4c5af2e17c42b644210bae0c6c88b',
#                'f7fbf477d959473b676fd5e53a243e51',
#                '0235947fe67f095afdce360a44932aa7',
#                '3f9349f4d58a0590c6575920163dbd45',
#                '252c00a8ee9a6dbb553e7166d460b4fe',
#                '06381599d71447f5c7c667e3e3106f88',
#                'e668a8d56be4b85ea0fe063c0511c617',
#                '494a1f3016417558967fc452e83604b0',
#                '5c135a6e2604d153a91e4fd757218b49',
#                '12a26501a9dd07618c85bd6f324237ed',
#                '1d13122682b8d57568e86741055d953b',
#                'f61c95aad79795bbe476c2a6692025d5']

simulations = ['8cda68f5b94f5a6294859908589917b6', 
               '1818dc5964723deb494a7c19de432430', 
               'e5ab6ba092eb6742d7f7d09fd5748825', 
               '3af40b2797084c2ba314e7278b656e54', 
               '0123cc2bb77249bc63523a0c43d80525', 
               '97b19f8a5c6ea36b607bc06516cf8a8e', 
               '6ea3d4a316646e051f7250c5dd72488b', 
               '9736fe4eff5dd042d99d6ca05197497a', 
               'b9a4181842479e1234d064800bc31e43', 
               'ff71c13df64d3f9f005d53374b257de5', 
               'e90e010be8da4d76e3ef53a4738c7e1d', 
               '3c917d2304ee6fc333f631fcfffc9b05', 
               'c3e14af4ed96c8a35fb44ea90efbe996', 
               'ea86066b7fd721942053f0fee2663dce', 
               'e9c87e165f69780dfa95fe1c0f4f0fd2']


color_s = ['blue', 'orange', 'green', 'red', 'peru', 'red', 'red', 'red', 'red', 'red']

g_A_s = np.array([0., 0.5, 1.])
apf_s = np.array([0., 0.05, 0.1, 0.2, 0.4])
n_gA = len(g_A_s)
n_apf = len(apf_s)

U_i = iteration.U_i
spread_active_inactive_states = iteration.spread_active_inactive_states
sum_active_inactive_states = iteration.sum_active_inactive_states
active = iteration.active
inactive = iteration.inactive


def sig_fun(r_i_k):
    """Activity of units"""
    rMax = np.max(r_i_k)
    sig_i_k = np.exp(beta * (r_i_k - rMax + U_i[:, None]))
    Z_i = spread_active_inactive_states.dot(
        sum_active_inactive_states.dot(sig_i_k))
    sig_i_k = sig_i_k/Z_i
    return sig_i_k


for ii in range(15):
    print(ii)
    key = simulations[ii]
    [_, durations] = file_handling.load_text(key+'/metrics_cue_%d.txt' % 0)
    ksi_i_mu, delta__ksi_i_mu__k, _, C_i_j = file_handling.load_network(key)

    coact_pos = file_handling.load_coact_pos(0, key)
    coact_neg = file_handling.load_coact_neg(0, key)

    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
        file_handling.load_parameters(key)

    patt_vect = delta__ksi_i_mu__k.astype(float) - a/S
    factor = np.sqrt(N*a*(1-a/S))
    patt_vect /= factor

    patt_corr = np.dot(np.transpose(patt_vect), patt_vect)
    M = np.dot(patt_vect, np.transpose(patt_vect))

    self_patt_corr = patt_corr[0, 0]

    # upper_bound = patt_corr/self_patt_corr
    upper_bound = np.zeros((p, p))
    lower_bound = np.zeros((p, p))

    n_set_r = 1
    random_set_r = rd.rand(N*(S+1), n_set_r)
    random_set_r[inactive, :] = -U
    random_sig_i_k = sig_fun(random_set_r)

    for nn in range(n_set_r):
        sig_i_k = random_sig_i_k[active, nn]
        sample = np.outer(np.dot(np.transpose(sig_i_k), patt_vect),
                          np.dot(np.transpose(patt_vect), sig_i_k))
        tmp_ind = sample > upper_bound
        upper_bound[tmp_ind] = sample[tmp_ind]
        tmp_ind = sample < lower_bound
        lower_bound[tmp_ind] = sample[tmp_ind]

    n_pairs = (p*(p-1))//2
    cpt = 0
    upper_bound_pair = np.zeros(n_pairs)
    corr_proj_pair = upper_bound_pair.copy()
    coact_pos_pair = upper_bound_pair.copy()
    coact_neg_pair = upper_bound_pair.copy()
    lower_bound_pair = upper_bound_pair.copy()

    points = np.zeros((2*n_pairs, 2))

    for patt1 in range(p):
        for patt2 in range(patt1+1, p):
            upper_bound_pair[cpt] = upper_bound[patt1, patt2]
            lower_bound_pair[cpt] = lower_bound[patt1, patt2]
            corr_proj_pair[cpt] = patt_corr[patt1, patt2]
            coact_pos_pair[cpt] = coact_pos[patt1, patt2]
            coact_neg_pair[cpt] = coact_neg[patt1, patt2]
            points[cpt, 0] = patt_corr[patt1, patt2]
            points[cpt, 1] = upper_bound[patt1, patt2]
            points[n_pairs+cpt, 0] = patt_corr[patt1, patt2]
            points[n_pairs+cpt, 1] = lower_bound[patt1, patt2]
            cpt += 1

    env = ConvexHull(points)

    ind_cor_sort = np.argsort(corr_proj_pair)

    ind_gA = [i for i in range(len(g_A_s)) if g_A_s[i] == g_A][0]
    ind_apf = [i for i in range(len(apf_s)) if apf_s[i] == a_pf][0]

    plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)
    jitterx = 0.1*np.max(corr_proj_pair)*rd.rand(n_pairs)
    jittery = 0.1*np.max(upper_bound)*rd.rand(n_pairs)
    plt.scatter(corr_proj_pair + jitterx, coact_pos_pair,
                color=color_s[ind_gA], s=1, alpha=1)
    plt.scatter(corr_proj_pair + jitterx, coact_neg_pair,
                color=color_s[ind_gA], s=1, alpha=1)
    if ind_gA == 0:
        plt.scatter(corr_proj_pair + jitterx, upper_bound_pair,
                    color='red', alpha=0.2, s=1)
        plt.plot(points[env.vertices, 0], points[env.vertices, 1], 'r--', lw=2)
        plt.scatter(corr_proj_pair + jitterx, lower_bound_pair,
                    color='red', alpha=0.2, s=1)

    # plt.plot(corr_proj_pair[ind_cor_sort],
    #          upper_bound_pair[ind_cor_sort], color=color_s[ind_gA])
    # plt.plot(corr_proj_pair[ind_cor_sort],
    #          -upper_bound_pair[ind_cor_sort],
    #          color=color_s[ind_gA])
