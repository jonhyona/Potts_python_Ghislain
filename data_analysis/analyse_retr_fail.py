import file_handling
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import correlations
from tqdm import tqdm

plt.ion()
plt.close('all')

simulations = ['b18e30bc89dbcb5bc2148fb9c6e0c51d']

alpha = 1
n_seeds = 3
key = simulations[0]

(dt, tSim, N, S, p, num_fact, p_fact,
 dzeta, a_pf,
 eps,
 f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
 tau_3_B, g_A,
 beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
 russo2008_mode, muted_prop) = file_handling.load_parameters(simulations[0])


def get_retrieved_seeds(key, n_seeds):
    retrieved = [[] for ii in range(n_seeds)]
    for kick_seed in range(n_seeds):
        retrieved[kick_seed] = file_handling.load_retrieved(kick_seed, key)
    return retrieved


def get_transition_times_seed(key, n_seeds):
    trans_times = [[] for ii in range(n_seeds)]
    for kick_seed in range(n_seeds):
        trans_times[kick_seed] = file_handling.load_transition_time(kick_seed, key)
    return trans_times


def flatten_diff_time(data):
    n_seeds = len(data)
    n_cues = len(data[0])
    res_prev = []
    res_folo = []
    for ind_seed in range(n_seeds):
        for ind_cue in range(n_cues):
            for ind_trans in range(len(data[ind_seed][ind_cue])):
                if ind_trans < len(data[ind_seed][ind_cue]) - 1:
                    res_prev.append(data[ind_seed][ind_cue][ind_trans])
                if ind_trans > 0:
                    res_folo.append(data[ind_seed][ind_cue][ind_trans])
    return np.array(res_prev), np.array(res_folo)


def get_pair_time(retrieved, trans_times):
    n_seeds = len(retrieved)
    n_cue = len(retrieved[0])
    res = [[[] for pattB in range(p)] for pattA in range(p)]
    for kick_seed in range(n_seeds):
        for ind_cue in range(n_cue):
            for ind_trans in range(3, len(retrieved[kick_seed][ind_cue])-1):
                pattA = retrieved[kick_seed][ind_cue][ind_trans]
                pattB = retrieved[kick_seed][ind_cue][ind_trans+1]
                diff_time = trans_times[kick_seed][ind_cue][ind_trans+1] \
                    - trans_times[kick_seed][ind_cue][ind_trans]
                res[pattA][pattB].append(diff_time)
    return res


def get_pair_mean_time(pair_time):
    res = np.zeros((p, p))
    for pattA in range(p):
        for pattB in range(p):
            if len(pair_time[pattA][pattB]) > 0:
                res[pattA][pattB] = np.mean(pair_time[pattA][pattB])
            else:
                res[pattA][pattB] = np.nan
    return np.reshape(res, p**2)


def get_pair_std(pair_time):
    res = np.zeros((p, p))
    for pattA in range(p):
        for pattB in range(p):
            if len(pair_time[pattA][pattB]) > 1:
                res[pattA][pattB] = np.std(pair_time[pattA][pattB])
            else:
                res[pattA][pattB] = np.nan
    return np.reshape(res, p**2)


trans_times = get_transition_times_seed(key, n_seeds)
trans_before, trans_after = flatten_diff_time(trans_times)
diff_times = trans_after - trans_before
# plt.figure('Test')
# plt.hist(diff_times, bins=150)
# plt.axvline(tau_2, 0, 1, label=r'$\tau_2$', color='k')
# plt.legend()

retrieved = get_retrieved_seeds(key, n_seeds)

pair_time = get_pair_time(retrieved, trans_times)

pair_mean_time = get_pair_mean_time(pair_time)
pair_std_time = get_pair_std(pair_time)

# plt.figure('Mean_time_hist')
# plt.hist(pair_mean_time, bins=100)
# plt.axvline(tau_2, 0, 1, label=r'$\tau_2$', color='k')


# plt.figure('Std_time_hist')
# plt.hist(pair_std_time, bins=100)
# plt.axvline(tau_2, 0, 1, label=r'$\tau_2$', color='k')

# plt.figure('QZFZEFQZE')
# plt.hist(pair_std_time/pair_mean_time, bins=100)
ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, _ = file_handling.load_network(key)
trans = []
trans_time = []
valid_transitions = []
invalid_transitions = []
valid_time = []
invalid_time = []
is_valid = []
tSnap = np.linspace(0, tSim, nSnap)
for kick_seed in range(1):
    for ind_cue in tqdm(range(1)):
        overlap = file_handling.load_evolution(ind_cue, kick_seed, key)
        retrieved = []
        prev_ret = -1
        waiting_validation = False
        valid = True
        patt_buffer = None
        for ind_t in range(nSnap-1):
            pattA = np.argmax(overlap[ind_t, :])
            max_m_mu = overlap[ind_t, pattA]
            overlap[ind_t, pattA] = -np.inf
            pattB = np.argmax(overlap[ind_t, :])
            max2_m_mu = overlap[ind_t, pattB]
            overlap[ind_t, pattA] = max_m_mu

            if pattA != prev_ret and max_m_mu > 0.5:
                trans.append((prev_ret, pattA))
                trans_time.append(tSnap[ind_t])
                if not valid:
                    invalid_transitions.append((prev_ret, blocker))
                    is_valid.append(False)
                prev_ret = pattA
                valid = False

            if not valid and max_m_mu > 0.5 and max_m_mu - max2_m_mu > 0.2:
                valid_transitions.append(trans[-1])
                is_valid.append(True)
                valid = True

            if not valid and max_m_mu > 0.5 and max_m_mu - max2_m_mu <= 0.2:
                blocker = pattB

plt.close('all')
plt.figure('evolution')
trans_time = file_handling.load_transition_time(0, key)[ind_cue]
retrieved = file_handling.load_retrieved(0, key)[ind_cue]
plt.plot(tSnap, overlap)
for ind_trans in range(len(trans_time)):
    plt.text(trans_time[ind_trans], 0.8, str(retrieved[ind_trans]))


plt.figure('Scatter')
C0, C1, C2 = [], [], []
for ind_trans in range(len(valid_transitions)):
    pair = valid_transitions[ind_trans]
    if pair[0] != -1:
        pattA = ksi_i_mu[:, pair[0]]
        pattB = ksi_i_mu[:, pair[1]]
        C0.append(correlations.active_inactive(pattA, pattB))
        C1.append(correlations.active_same_state(pattA, pattB))
        C2.append(correlations.active_diff_state(pattA, pattB))
        if correlations.active_same_state(pattA, pattB) > 0.8:
            print(pair[0], pair[1])

plt.scatter(C1, C0, color='b')

C0, C1, C2 = [], [], []
for ind_trans in range(len(invalid_transitions)):
    pair = invalid_transitions[ind_trans]
    if pair[0] != -1:
        pattA = ksi_i_mu[:, pair[0]]
        pattB = ksi_i_mu[:, pair[1]]
        C0.append(correlations.active_inactive(pattA, pattB))
        C1.append(correlations.active_same_state(pattA, pattB))
        C2.append(correlations.active_diff_state(pattA, pattB))

plt.scatter(C1, C0, color='r')

cpt_valid = -1
cpt_invalid = 0
for ind in range(len(is_valid)):
    if not is_valid[ind]:
        pattA = invalid_transitions[cpt_invalid][0]
        pattB = invalid_transitions[cpt_invalid][1]
        pattC = valid_transitions[cpt_valid][1]
        if pattA == pattC:
            pattC = invalid_transitions[cpt_valid][0]
        ksiA = ksi_i_mu[:, pattA]
        ksiB = ksi_i_mu[:, pattB]
        ksiC = ksi_i_mu[:, pattC]
        CAC = np.sum(np.logical_and(ksiA == ksiC, 1-(ksiA == S)))
        CBC = np.sum(np.logical_and(ksiB == ksiC, 1-(ksiB == S)))
        print(pattA, pattB, pattC, CAC, CBC, np.abs(CAC - CBC))
        cpt_invalid += 1
    else:
        cpt_valid += 1

for ind in range(len(valid_transitions)):
    pattC = valid_transitions[ind][1]
    for pattA in range(p):
        if pattA != pattC:
            for pattB in range(p):
                if pattB not in (pattA, pattC):
                    ksiA = ksi_i_mu[:, pattA]
                    ksiB = ksi_i_mu[:, pattB]
                    ksiC = ksi_i_mu[:, pattC]
                    CAC = np.sum(np.logical_and(ksiA == ksiC, 1-(ksiA == S)))
                    CBC = np.sum(np.logical_and(ksiB == ksiC, 1-(ksiB == S)))
                    print(pattA, pattB, pattC, CAC, CBC, np.abs(CAC - CBC))
                    
            
# print(invalid_transitions)
