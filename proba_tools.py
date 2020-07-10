import numpy as np
import file_handling
from tqdm import tqdm
import sparse


def trio_prob_table(retrieved, key):
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode, muted_prop) = \
         file_handling.load_parameters(key)
    n_seeds = len(retrieved)

    num_ABC = np.zeros((p, p, p), dtype=int)
    num_AB = np.zeros((p, p), dtype=int)
    num_A = np.zeros(p, dtype=int)
    num_B = num_A.copy()

    p_B_ABC = np.nan*np.ones((p, p, p), dtype=float)
    p_AB_ABC = np.nan*np.ones((p, p, p), dtype=float)
    p_A = np.nan*np.ones(p, dtype=float)
    p_B = np.nan*np.ones(p, dtype=float)

    for kick_seed in range(n_seeds):
        for cue_ind in range(p):
            if len(retrieved[kick_seed][cue_ind]) >= 3:
                # print(len(retrieved[kick_seed][cue_ind]))
                duration = len(retrieved[kick_seed][cue_ind])
                if cue_ind != retrieved[kick_seed][cue_ind][0]:
                    duration += 1
                # ind_max[cue_ind] = duration
                sequence = []
                if cue_ind != retrieved[kick_seed][cue_ind][0]:
                    sequence.append(cue_ind)
                sequence += retrieved[kick_seed][cue_ind]
                sequence = sequence[3:]

                for ind_trans in range(len(sequence)-2):
                    pattA = sequence[ind_trans]
                    pattB = sequence[ind_trans+1]
                    pattC = sequence[ind_trans+2]
                    num_AB[pattA, pattB] += 1
                    num_A[pattA] += 1
                    num_B[pattB] += 1
                    num_ABC[pattA, pattB, pattC] += 1

    p_A = num_A / np.sum(num_A)
    p_B = num_B / np.sum(num_B)
    p_AB = num_AB / np.sum(num_B)
    occuring_B = num_B != 0
    # print(num_B)
    # print(num_B.shape)
    p_B_ABC[:, occuring_B, :] = num_ABC[:, occuring_B, :] \
        / num_B[None, occuring_B, None]
    occuring_AB = num_AB != 0
    p_AB_ABC[occuring_AB, :] = num_ABC[occuring_AB, :] \
        / num_AB[occuring_AB, None]
    p_ABC = num_ABC / np.sum(num_B)
    return num_A, p_A, num_B, p_B, num_AB, p_AB, num_ABC, p_ABC, p_B_ABC, \
        p_AB_ABC


def build_trans_tables(retrieved, key, L):
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode, muted_prop) = \
         file_handling.load_parameters(key)

    n_seeds = len(retrieved)
    num_tables = [[] for order in range(L+1)]
    proba_tables = [[] for order in range(L+1)]
    print('Table creation')
    for order in tqdm(range(L)):
        chain_length = order + 1
        num_tables[order] = sparse.DOK(shape=tuple([p for ii in
                                                    range(chain_length)]))
    print('Fill num_tables')
    for kick_seed in tqdm(range(n_seeds)):
        for cue_ind in range(p):
            if len(retrieved[kick_seed][cue_ind]) >= 3:
                # print(len(retrieved[kick_seed][cue_ind]))
                sequence = []
                sequence = retrieved[kick_seed][cue_ind][3:]

                for ind_trans in range(len(sequence)-L-1):
                    trans_string = sequence[ind_trans: ind_trans + L+1]
                    for order in range(L):
                        string = trans_string[: order + 1]
                        num_tables[order][tuple(string)] += 1
    print('Table conversion')
    for order in tqdm(range(L)):
        num_tables[order] = sparse.COO(num_tables[order])
        proba_tables[order] = num_tables[order] / num_tables[order].sum()
    return num_tables, proba_tables


def condi_prob(Z, XY, XYZ, proba_tables):
    p_XYZ = proba_tables[len(XYZ)-1][tuple(XYZ)]
    p_XY = proba_tables[len(XY)-1][tuple(XY)]
    # print(p_XYZ, p_XY)
    if p_XY > 0:
        return p_XYZ/p_XY
    return np.nan
