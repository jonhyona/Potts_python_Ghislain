import file_handling
import numpy as np
import matplotlib.pyplot as plt
import proba_tools
import matplotlib.cm as color_map
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import os

plt.ion()
plt.close('all')

# simulations = ['f30d8a2438252005f6a9190c239c01c1']
simulations = ['9e0fbd728bd38ee6eb130d85f35faa9a']

key = simulations[0]
n_seeds = 1
L = 3
retrieved = file_handling.load_retrieved_several(n_seeds, key)
num_tables, proba_tables = proba_tools.build_trans_tables(retrieved, key, L)

alpha = 1
p_min = 1e-3
g_min = 9e-4
alpha = 17.5
r = 1.6

(dt, tSim, N, S, p, num_fact, p_fact,
 dzeta, a_pf,
 eps,
 f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
 tau_3_B, g_A,
 beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
 russo2008_mode, muted_prop) = file_handling.load_parameters(simulations[0])

Y = 0
relevant_seq = [[] for order in range(0, L)]
relevant_seq[0] = [[Y]]


for order in range(2, L):
    for ind_XYZ in range(num_tables[order].coords.shape[1]):
        XYZ = num_tables[order].coords[:, ind_XYZ]
        X = XYZ[0]
        Z = XYZ[-1]
        Y = XYZ[1:-1]
        XY = XYZ[:-1]
        YZ = XYZ[1:]
        p_XY = proba_tables[order-1][tuple(XY)]
        if p_XY > p_min:
            informative = False
            p_XY_Z = proba_tools.condi_prob(Z, XY, XYZ, proba_tables)
            if p_XY_Z >= (1+alpha)*g_min:
                # print(p_XY_Z)

                p_Y_Z = proba_tools.condi_prob(Z, Y, YZ, proba_tables)
                # print(X, Z, p_XY_Z, p_Y_Z)
                if p_XY_Z / p_Y_Z >= r or p_XY_Z / p_Y_Z <= 1/r:
                    informative = True
            if informative:
                relevant_seq[order].append(XYZ)


def find_saddle(relevant_seq):
    seq = relevant_seq[2]
    by_second = [[[] for nu in range(p)] for mu in range(p)]
    saddle = []
    for XYZ in seq:
        by_second[XYZ[1]][XYZ[2]].append(XYZ)
    for mu in range(p):
        for nu in range(p):
            if len(by_second[mu][nu]) >= 2:
                saddle += by_second[mu][nu]
    return saddle

width = 2


def construct_graph(saddles):
    color_fun = color_map.plasma
    G = nx.MultiDiGraph()
    prev_Y = saddles[0][1]
    for ind_saddle in range(len(saddles)):
        XYZ = saddles[ind_saddle]
        if (XYZ == np.array([161, 29, 91])).all():
            print(XYZ)
        X = XYZ[0]
        Y = XYZ[1]
        XY = XYZ[:-1]
        Z = XYZ[2]
        YZ = XYZ[1:]
        if Y != prev_Y:
            write_dot(G, 'graph/my_graph%d.dot' % prev_Y)
            os.system('dot -Tjpg -o graph/my_graph%d.jpg graph/my_graph%d.dot' % (prev_Y, prev_Y))
            G = nx.MultiDiGraph()
            prev_Y = Y
        # print(XYZ)
        G.add_nodes_from(XYZ)
        rgba_color = (256*np.array(color_fun(X/p))).astype(int)

        if not G.has_edge(X, Y):
            G.add_edge(X, Y, color='#%02x%02x%02x' %
                       tuple(rgba_color[: -1]), penwidth=width)

        if not G.has_edge(Y, Z):
            G.add_edge(Y, Z, color='black', label='%.2f' %
                       proba_tools.condi_prob(Z, [Y], YZ,
                                              proba_tables), penwidth=width)
        G.add_edge(Y, Z, color='#%02x%02x%02x' %
                   tuple(rgba_color[:-1]), label='%.2f' %
                   proba_tools.condi_prob(Z, XY, XYZ, proba_tables),
                   penwidth=width)
        # print(proba_tools.condi_prob(Z, XY, XYZ, proba_tables))


            # G[Y][Z][-1]['color'] = '#%02x%02x%02x' % tuple(rgba_color[:-1])

        # G[X][Y]['color'] = 'black'
        # G[Y][Z]['color'] = 'black'
        # G[X][Y]['label'] = '%.2f' % proba_tools.condi_prob(Z, [Y], YZ,
        #                                                    proba_tables)
        # G[Y][Z]['label'] = '%.2f' % proba_tools.condi_prob(Z, XY, XYZ,
        #                                                    proba_tables)
    write_dot(G, 'graph/my_graph%d.dot' % Y)
    os.system('dot -Tjpg -o graph/my_graph%d.jpg graph/my_graph%d.dot' % (Y, Y))


saddles = find_saddle(relevant_seq)
construct_graph(saddles)


def issublist(small_list, big_list):
    for ii in range(len(big_list)-len(small_list)):
        if big_list[ii: ii+len(small_list)] == small_list:
            return True
    return False


print("Sublist check")
for cue in range(p):
    for saddle in saddles:
        if issublist(saddle.tolist(), retrieved[0][cue][:10]):
            if saddle[1] == 124:
                print(cue, saddle)
