import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import seaborn as sns
# Local modules
import file_handling
import networkx as nx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D


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

# simualtions = ['139a320d8afab59bcb18d59268071d94', 
#                '8d135d1215719ea60f9b7d297f1e9aeb', 
#                'a63c5077ffe56f59cde4dc33b7a7ac82', 
#                'ed63915a9b091c1018d50a3d99d48b41', 
#                '0786d89b52b48b48396d7f7c176fbe55', 
#                '9e046fe0ec70f37dc196979b42954507', 
#                '3373333f578dc83ba5aa6cd518b07a41', 
#                'e76fd063be5300c5ae1c37a57b850d7d', 
#                'e96993c97f950bdabf6c22d81396027d', 
#                'd158c4222b3dc4d3995869bccdc2f1a3', 
#                'f961b5cfb447cb3227dd2b5f5c71c4c1', 
#                '89c8b67bcd8c40231b7abc89dd30a51e']


def find_neighbors(key):
    (dt, tSim, N, S, p, num_fact, p_fact,
     dzeta, a_pf,
     eps,
     f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
     tau_3_B, g_A,
     beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
     russo2008_mode) = file_handling.load_parameters(key)

    graph_all = nx.Graph()
    graph_high = nx.Graph()
    graph_low = nx.Graph()

    num_trans_all = np.zeros((p, p), dtype=int)
    num_trans_high = np.zeros((p, p), dtype=int)
    num_trans_low = np.zeros((p, p), dtype=int)

    retrieved_saved = file_handling.load_retrieved(key)
    lamb = file_handling.load_overlap(key)

    neighbors = [[] for pat in range(p)]

    for cue_ind in range(p):
        if len(retrieved_saved[cue_ind]) >= 3:
            # print(len(retrieved_saved[cue_ind]))
            sequence = []
            if cue_ind != retrieved_saved[cue_ind][0]:
                sequence.append(cue_ind)
            sequence += retrieved_saved[cue_ind]
            sequence = sequence[3:]

            for ind_trans in range(len(sequence)-1):
                patt1 = sequence[ind_trans]
                patt2 = sequence[ind_trans+1]

                num_trans_all[patt1, patt2] += 1
                num_trans_all[patt2, patt1] += 1

                if lamb[cue_ind][ind_trans+1] >= 0.825:
                    num_trans_high[patt1, patt2] += 1
                    num_trans_high[patt2, patt1] += 1
                else:
                    num_trans_low[patt1, patt2] += 1
                    num_trans_low[patt2, patt1] += 1

                if patt2 not in neighbors[patt1]:
                    neighbors[patt1].append(patt2)

    for patt1 in range(p):
        for patt2 in range(patt1, p):
            if num_trans[patt1, patt2]:
                graph_all.add_edge(patt1, patt2,
                                   weight=num_trans[patt1, patt2])
            if num_trans_high[patt1, patt2]:
                graph_high.add_edge(patt1, patt2,
                                    weight=num_trans_high[patt1,
                                                          patt2])
            if num_trans_low[patt1, patt2]:
                graph_low.add_edge(patt1, patt2,
                                   weight=num_trans_low[patt1, patt2])

    return neighbors, graph_all, num_trans_all, graph_low, num_trans_low, \
        graph_high, num_trans_high


def count_neigbours(neighbors):
    counters = [0 for patt in range(len(neighbors))]
    for patt in range(len(neighbors)):
        counters[patt] = len(neighbors[patt])
    sns.distplot(counters)


g_A_s = [0., 0.5, 1.]
apf_s = [0., 0.05, 0.1, 0.2]
n_gA = len(g_A_s)
n_apf = len(apf_s)


plt.figure('neighbors_counter')
for ii in range(15):
    key = simulations[ii]
    print(ii)
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
        file_handling.load_parameters(key)
    if a_pf <= 0.3:
        ind_gA = [i for i in range(len(g_A_s)) if g_A_s[i] == g_A][0]
        ind_apf = [i for i in range(len(apf_s)) if apf_s[i] == a_pf][0]

        plt.figure('Neighbors_count')
        plt.subplot(3, 2, ind_apf+1)
        neighbors, graph_all, num_trans_all, graph_low, num_trans_low, \
            graph_high, num_trans_high = find_neighbors(key)
        count_neigbours(neighbors)
        plt.title(r"$a_{pf}$=%.2f" % a_pf)
        plt.legend([r"$g_A$=0.0, $w$=1.", r"$g_A$=0.5, $w$=1.2", r"$g_A$=1.0, $w$=1.4"])
        plt.xlabel('Number of neighbors')
        plt.ylabel('Density')

        plt.figure('Graph_high ')
        plt.subplot(3,2,ii//3+1)
        def trans_width(x): return 1
        widths = np.array(list(nx.get_edge_attributes(graph_high, 'weight').values()))
        norm_width = trans_width(widths)
        plt.title(r'$a_{pf}$=%.2f, $g_A$=%.1f, $w$=%.1f' % (a_pf, g_A, w))
        nx.draw_spring(graph_high, width=norm_width, node_size=50)

        xx = np.arange(0, p, 1)
        yy = np.logspace(0, np.log10(np.max(num_trans_all)), 20)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = np.zeros((len(yy), len(xx)))
        max_non_null = np.zeros(p)

        for ii in range(len(xx)):
            for jj in range(len(yy)):
                num_trans_all_p = np.sum(num_trans_all[ii, :] >= yy[jj])
                ZZ[jj, ii] = num_trans_all_p
                # print(num_trans_all_p)
                # if num_trans_all_p == 0:
                #     ZZ[jj, ii] = np.nan
                if num_trans_all_p >= 1:
                    max_non_null[ii] = jj

        # for patt1 in range(p):
        #     for patt2 in range(patt1, p):
        #         if num_trans_all[patt1, patt2] == np.max(num_trans_all):
        #             # print(patt1, patt2)

        ind_gA = [i for i in range(len(g_A_s)) if g_A_s[i] == g_A][0]
        ind_apf = [i for i in range(len(apf_s)) if apf_s[i] == a_pf][0]

        indexes = np.argsort(max_non_null)
        # print(indexes)
        # xx = xx[indexes]
        XX, YY = np.meshgrid(xx, yy)
        ZZ = ZZ[:, indexes]
        plt.figure('Distribution_distribution_gA%.1f' % g_A)
        plt.subplot(2, 2, ind_apf + 1)
        plt.pcolor(XX, YY, ZZ, norm=colors.LogNorm(vmin=1, vmax=200))
        plt.colorbar()
        plt.title(r'$a_{pf}$=%.2f, $g_A$=%.1f' % (a_pf, g_A))
        plt.xlabel('Reordered patterns')
        plt.ylabel('Number of transition')
        plt.ylim(1, np.max(num_trans_all))
        # plt.title('Number of neighbors with sufficient number of transitions \n' +r'$a_{pf}$=%.2f, $g_A$=%.1f, $w$=%.1f' % (a_pf, g_A, w))
        plt.yscale('log')

plt.tight_layout()
plt.show()

# plt.figure('activation')
# xx = np.linspace(1., np.max(widths), 200)
# plt.plot(xx, trans_width(xx))

for patt1 in range(p):
    for patt2 in range(p):
        if num_trans_all[patt1, patt2] == np.max(num_trans_all):
            print(patt1, patt2)
