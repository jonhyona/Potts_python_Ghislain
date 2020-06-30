import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import seaborn as sns
# Local modules
import file_handling
import networkx as nx
from scipy.optimize import curve_fit
from scipy.special import erf
from statistics import median
import scipy.stats as st

import warnings
warnings.filterwarnings("ignore")


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

color_s = ['blue', 'orange', 'green', 'red', 'peru', 'red', 'red', 'red', 'red', 'red']


def find_ind_max(sim_list):
    ind_max = np.zeros(p, dtype=int)
    for ind_key in range(len(sim_list)):
        key = sim_list[ind_key]
        retrieved_saved = file_handling.load_retrieved(key)

        for cue_ind in range(p):
            duration = len(retrieved_saved[cue_ind])
            if cue_ind != retrieved_saved[cue_ind][0]:
                duration += 1
            if ind_max[cue_ind] != 0:
                ind_max[cue_ind] = min(ind_max[cue_ind], duration)
            else:
                ind_max[cue_ind] = duration
    return ind_max


try:
    ind_max
except NameError as ex:
    ind_max = find_ind_max(simulations)


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

    threshold = median([lamb[ind_cue][trans] for ind_cue in range(p) for trans in range(len(lamb[ind_cue]))])

    neighbors = [[] for pat in range(p)]

    for cue_ind in range(p):
        if len(retrieved_saved[cue_ind][:ind_max[cue_ind]]) >= 3:
            # print(len(retrieved_saved[cue_ind]))
            duration = len(retrieved_saved[cue_ind])
            if cue_ind != retrieved_saved[cue_ind][0]:
                duration += 1
            # ind_max[cue_ind] = duration
            sequence = []
            if cue_ind != retrieved_saved[cue_ind][0]:
                sequence.append(cue_ind)
            sequence += retrieved_saved[cue_ind]
            sequence = sequence[3:ind_max[cue_ind]]

            for ind_trans in range(len(sequence)-1):
                patt1 = sequence[ind_trans]
                patt2 = sequence[ind_trans+1]

                num_trans_all[patt1, patt2] += 1

                if lamb[cue_ind][ind_trans+1] >= threshold:
                    num_trans_high[patt1, patt2] += 1
                else:
                    num_trans_low[patt1, patt2] += 1

                # if True:
                if patt2 not in neighbors[patt1]:
                    neighbors[patt1].append(patt2)

    for patt1 in range(p):
        for patt2 in range(p):
            if patt1 != patt2:
                if num_trans_all[patt1, patt2]:
                    graph_all.add_edge(patt1, patt2,
                                       weight=num_trans_all[patt1,
                                                            patt2])
                if num_trans_high[patt1, patt2]:
                    graph_high.add_edge(patt1, patt2,
                                        weight=num_trans_high[patt1,
                                                              patt2])
                if num_trans_low[patt1, patt2]:
                    graph_low.add_edge(patt1, patt2,
                                       weight=num_trans_low[patt1,
                                                            patt2])

    return neighbors, graph_all, num_trans_all, graph_low, num_trans_low, \
        graph_high, num_trans_high, threshold


def count_neighbors(neighbors):
    counters = [0 for patt in range(len(neighbors))]
    for patt in range(len(neighbors)):
        counters[patt] = len(neighbors[patt])
    # bins = np.arange(np.min(counters)-0.5, np.max(counters)+1.5, 1)
    sns.distplot(counters)


g_A_s = [0., 0.5, 1.]
apf_s = [0., 0.05, 0.1, 0.2]
n_gA = len(g_A_s)
n_apf = len(apf_s)

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

        neighbors, graph_all, num_trans_all, graph_low, num_trans_low, \
            graph_high, num_trans_high, threshold = find_neighbors(key)

        # plt.figure('Neighbors_count')
        # plt.subplot(2, 2, ind_apf+1)
        # count_neighbors(neighbors)
        # plt.title(r"$a_{pf}$=%.2f" % a_pf)
        # plt.xlabel('Number of neighbors')
        # plt.ylabel('Density')
        # plt.tight_layout()

        # def trans_width(x): return 1

        # plt.figure('Graph_high')
        # plt.subplot(2, 2, ii//3+1)
        # widths = np.array(list(nx.get_edge_attributes(graph_high, 'weight').values()))
        # norm_width = trans_width(widths)
        # nx.draw_spring(graph_high, width=norm_width, node_size=50)
        # # plt.tight_layout()
        # plt.title(r'$a_{pf}$=%.2f, $g_A$=%.1f, $w$=%.1f, $\lambda > \lambda_{th}$=%.2f' % (a_pf, g_A, w, threshold))
        # plt.figure('Graph_low')
        # plt.subplot(2,2,ii//3+1)
        # widths = np.array(list(nx.get_edge_attributes(graph_low, 'weight').values()))
        # norm_width = trans_width(widths)
        # plt.title(r'$a_{pf}$=%.2f, $g_A$=%.1f, $w$=%.1f, $\lambda < \lambda_{th}$=%.2f' % (a_pf, g_A, w, threshold))
        # nx.draw_spring(graph_low, width=norm_width, node_size=50)
        # # plt.tight_layout()

        # plt.figure('Graph_all')
        # plt.subplot(2,2,ii//3+1)
        # widths = np.array(list(nx.get_edge_attributes(graph_all, 'weight').values()))
        # norm_width = trans_width(widths)
        # plt.title(r'$a_{pf}$=%.2f, $g_A$=%.1f, $w$=%.1f' % (a_pf, g_A, w))
        # nx.draw_spring(graph_all, width=norm_width, node_size=50)
        # plt.tight_layout()

        # xx = np.arange(0, p, 1)
        # # yy = np.logspace(0, np.log10(np.max(num_trans_all)+2), 20)
        # yy1 = np.array(range(1, np.max(num_trans_all)+1))
        # XX, YY1 = np.meshgrid(xx, yy1)
        # cumul = np.zeros((len(yy1), len(xx)), dtype=int)
        # distrib = cumul.copy()
        # max_yy1 = np.zeros(p)
        # max_distrib = np.zeros(p)
        # max_cumul = np.zeros(p)

        # for ii in range(len(xx)):
        #     for jj in range(len(yy1)):
        #         num_trans_all_p = np.sum(num_trans_all[ii, :] >= yy1[jj])
        #         cumul[jj, ii] = num_trans_all_p
        #         if jj != len(yy1)-1:
        #             distrib[jj, ii] = \
        #                 np.sum(np.logical_and(num_trans_all[ii, :] >=
        #                                       yy1[jj],
        #                                       num_trans_all[ii, :] <
        #                                       yy1[jj+1]))

        #         if num_trans_all_p >= 1:
        #             max_yy1[ii] = max(yy1[jj], max_yy1[ii])
        #             max_distrib[ii] = max(distrib[jj, ii], max_distrib[ii])
        #             max_cumul[ii] = max(cumul[jj, ii], max_distrib[ii])

        # indices_sort1 = np.argsort(max_yy1)
        # XX, YY1 = np.meshgrid(xx, yy1)
        # plt.figure('Distribution_distribution_cumul_gA%.1f' % g_A)
        # plt.subplot(2, 2, ind_apf + 1)
        # plt.pcolor(XX, YY1, cumul[:, indices_sort1],
        #            norm=colors.LogNorm(vmin=1, vmax=200))
        # plt.plot(xx, max_yy1[indices_sort1], color='orange')
        # plt.colorbar()
        # plt.title(r'$a_{pf}$=%.2f, $g_A$=%.1f' % (a_pf, g_A))
        # plt.xlabel('Reordered patterns')
        # plt.ylabel('Number of transition')
        # plt.ylim(0.8, 1e3)
        # plt.yscale('log')
        # plt.tight_layout()

        # yy2 = np.array(range(p))
        # XX, YY2 = np.meshgrid(xx, yy2)
        # data = np.zeros((len(yy2), len(xx)), dtype=int)
        # data_cumul = data.copy()
        # max_non_null_data = np.zeros(p)
        # max_non_null_data_cumul = max_non_null_data.copy()
        # for ii in range(len(xx)):
        #     for jj in range(len(yy2)):
        #         cond = distrib[:, ii] == yy2[jj]
        #         if cond.any():
        #             data[jj, ii] = np.max(yy1[cond])
        #         if data[jj, ii] >= 0.5:
        #             max_non_null_data[ii] = max(yy2[jj], max_non_null_data[ii])
        #         tmp_trans = sorted(num_trans_all[ii])
        #         data_cumul[jj, ii] = tmp_trans[-xx[jj]]
        #         if data_cumul[jj, ii] != 0:
        #             max_non_null_data_cumul[ii] = max(yy2[jj],
        #                                               max_non_null_data_cumul[ii])

        # indices_sort2 = np.argsort(max_non_null_data_cumul)

        # plt.figure('Reverse_planar_cumul_gA%.1f' % g_A)
        # plt.subplot(2, 2, ind_apf + 1)
        # plt.pcolor(XX, YY2, data_cumul[:, indices_sort2],
        #            norm=colors.LogNorm(vmin=1, vmax=1e3))
        # plt.plot(xx, max_non_null_data_cumul[indices_sort2], color='orange')
        # plt.colorbar()
        # plt.title(r'$a_{pf}$=%.2f, $g_A$=%.1f' % (a_pf, g_A))
        # plt.xlabel('Reordered patterns')
        # plt.ylabel('Number of patterns')
        # plt.ylim(0.8, 200)
        # plt.yscale('log')
        # plt.tight_layout()

        fig = plt.figure('Fit_log_normal')
        ax = plt.subplot(2, 2, ind_apf+1)

        def fit_function(x, A, n0, s, slope):
            # return A/s/x/np.sqrt(2*np.pi)*np.exp(-(np.log(x)-n0)**2/2/s**2)*np.exp(-np.log(x)*slope)
            return A/x*np.exp(-np.log(x)*slope)

        data = np.zeros(p*(p-1))
        cpt = 0
        for patt1 in range(p):
            for patt2 in range(p):
                if patt1 != patt2:
                    data[cpt] = num_trans_all[patt1, patt2]
                    cpt += 1
        data = data[data != 0]

        est_s = np.log(1+np.var(data)/np.mean(data)**2)
        est_mu = np.log(np.mean(data)) - 1/2*est_s**2

        data_entries, bins = np.histogram(data,
                                          bins=np.arange(np.min(data)-0.5,
                                                         np.max(data)+1.5),
                                          density=True)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in
                                range(len(bins)-1)])

        test_distrib = st.norm
        data_log = np.log(data)
        y0 = np.mean(data_log)
        log_sig = np.std(data_log)
        fit_success = True
        try:
            popt, pcov = curve_fit(fit_function, xdata=binscenters,
                                   ydata=data_entries,
                                   p0=[data_entries[0], -2, 1.5, 1])
        except RuntimeError as ex:
            fit_success = False

        print(a_pf, g_A, popt)

        xspace = np.arange(1, np.max(binscenters), 1)
        xspace_log = np.arange(0, 4, 0.1)

        if fit_success:
            plt.plot(xspace, fit_function(xspace, *popt),
                     color=color_s[ind_gA], linewidth=2.5)
            # plt.plot(xspace, test_distrib.pdf(xspace, 1.5, 0, 0.1),
            #          color=color_s[ind_gA], linewidth=2.5)
            print('Success')
        plt.bar(binscenters, data_entries, width=bins[1] - bins[0],
                color=color_s[ind_gA], label=r'$g_A$=%.1f' % g_A, alpha=.2)
        plt.xlim(np.min(binscenters), np.max(binscenters))
        plt.yscale('log')
        plt.ylim(1e-4, 1)
        plt.xscale('log')
        plt.xlim(0, 1e3)
        plt.xlabel('Number of transitions')
        plt.ylabel('Occurences')
        plt.title(r"$a_{pf}$=%.2f" % a_pf)
        plt.legend()
        plt.tight_layout()

        # plt.figure('Fit_log_normal_cum')
        # ax = plt.subplot(2, 2, ind_apf+1)

        # data_entries, bins = np.histogram(data,
        #                                   bins=np.arange(np.min(data)-0.5,
        #                                                  np.max(data)+1.5))
        # binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in
        #                         range(len(bins)-1)])

        # data_entries = np.cumsum(data_entries[::-1])[::-1]

        # def fit_function(x, A, n0, s):
        #     return A*(1/2-1/2*erf((np.log(x)-n0)/s/np.sqrt(2)))

        # fit_success = True
        # try:
        #     popt, pcov = curve_fit(fit_function, xdata=binscenters,
        #                            ydata=data_entries,
        #                            p0=[data_entries[0], -2, 1.5])
        # except RuntimeError as ex:
        #     fit_success = False
        # xspace = np.arange(1, np.max(binscenters), 1)
        # if fit_success:
        #     plt.plot(xspace, fit_function(xspace, *popt),
        #              color=color_s[ind_gA], linewidth=2.5)
        # plt.bar(binscenters, data_entries, width=bins[1] - bins[0],
        #         color=color_s[ind_gA], label=r'$g_A$=%.1f' % g_A, alpha=.2)
        # plt.xlim(np.min(binscenters), np.max(binscenters))
        # plt.yscale('log')
        # plt.ylim(1, 1e4)
        # plt.xscale('log')
        # plt.xlim(1, 1e3)
        # plt.xlabel('Number of transitions')
        # plt.ylabel('Occurences')
        # plt.title(r"$a_{pf}$=%.2f" % a_pf)
        # plt.legend()
        # plt.tight_layout()
