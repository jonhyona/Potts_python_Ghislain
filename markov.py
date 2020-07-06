import file_handling
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

plt.ion()
plt.close('all')

simulations = ['f30d8a2438252005f6a9190c239c01c1']

alpha = 1
n_seeds = 6
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


def trio_prob_table(retrieved):
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


def random_eq(retrieved, n_seeds):
    random_retrieved = copy.deepcopy(retrieved)
    shuffled_retrieved = copy.deepcopy(retrieved.copy())
    for kick_seed in range(n_seeds):
        for cue_ind in range(p):
            random_retrieved[kick_seed][cue_ind] = list(rd.randint(0, p,
                                                        len(retrieved[kick_seed][cue_ind])))
            if retrieved[kick_seed][cue_ind] == cue_ind:
                random_retrieved[kick_seed][cue_ind][0] = cue_ind
                rd.shuffle(shuffled_retrieved[kick_seed][cue_ind][1:])
            else:
                rd.shuffle(shuffled_retrieved[kick_seed][cue_ind])
    return random_retrieved, shuffled_retrieved



retrieved = get_retrieved_seeds(key, n_seeds)
(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
 cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
 g, random_seed, p_0, n_p, nSnap, russo2008_mode, muted_prop) = \
     file_handling.load_parameters(key)
random_retrieved, shuffled_retrieved = random_eq(retrieved, n_seeds)

num_A, p_A, num_B, p_B, num_AB, p_AB, num_ABC, p_ABC, p_B_ABC, \
        p_AB_ABC = trio_prob_table(retrieved)

proba_table = np.zeros((p, p))
occuring_A = num_A != 0
proba_table[occuring_A, :] = num_AB[occuring_A, :] / num_A[occuring_A, None]
retrieved_markov = copy.deepcopy(retrieved)

for kick_seed in range(n_seeds):
    for cue_ind in range(p):
        if len(retrieved[kick_seed][cue_ind]) >= 3:
            # print(len(retrieved[kick_seed][cue_ind]))
            duration = len(retrieved[kick_seed][cue_ind])
            if cue_ind != retrieved[kick_seed][cue_ind][0]:
                duration += 1
            retrieved_markov[kick_seed][cue_ind].append(cue_ind)
        if occuring_A[cue_ind]:
                prev_mu = cue_ind
                for ind_trans in range(duration-1):
                    prev_mu = rd.choice(np.array(range(p)), 1,
                                        p=proba_table[prev_mu, :].ravel())[0]
                    retrieved_markov[kick_seed][cue_ind].append(prev_mu)


num_A_rand, p_A_rand, num_B_rand, p_B_rand, num_AB_rand, p_AB_rand, \
    num_ABC_rand, p_ABC_rand, p_B_ABC_rand, p_AB_ABC_rand = \
    trio_prob_table(random_retrieved)
num_A_shuf, p_A_shuf, num_B_shuf, p_B_shuf, num_AB_shuf, p_AB_shuf, \
    num_ABC_shuf, p_ABC_shuf, p_B_ABC_shuf, p_AB_ABC_shuf = \
    trio_prob_table(shuffled_retrieved)
num_A_markov, p_A_markov, num_B_markov, p_B_markov, num_AB_markov, p_AB_markov, \
    num_ABC_markov, p_ABC_markov, p_B_ABC_markov, p_AB_ABC_markov = \
    trio_prob_table(retrieved_markov)

est = np.multiply(p_ABC, p_B[None, :, None]) \
    - np.multiply(p_AB[:, :, None], p_AB[None, :, :])
est_rand = np.multiply(p_ABC_rand, p_B_rand[None, :, None]) \
    - np.multiply(p_AB_rand[:, :, None], p_AB_rand[None, :, :])
est_shuf = np.multiply(p_ABC_shuf, p_B_shuf[None, :, None]) \
    - np.multiply(p_AB_shuf[:, :, None], p_AB_shuf[None, :, :])
est_markov = np.multiply(p_ABC_markov, p_B_markov[None, :, None]) \
    - np.multiply(p_AB_markov[:, :, None], p_AB_markov[None, :, :])

kde = False
threshold = np.inf
kde_kws = {"bw": 1e-6, 'kernel': 'tri'}

plot_est = np.reshape(est, p**3)
plot_est = plot_est[np.abs(plot_est) < threshold]
plot_est_markov = np.reshape(est_markov, p**3)
plot_est_markov = plot_est_markov[np.abs(plot_est_markov) < threshold]

plt.close('markov_test_ABC')
plt.figure('markov_test_ABC')
sns.distplot(plot_est, label='Latching', kde=kde, norm_hist=True, kde_kws=kde_kws)
# sns.distplot(np.reshape(est_rand, p**2), label='Random', kde=kde)
# sns.distplot(np.reshape(est_shuf, p**2), label='Shuffled', kde=kde)
sns.distplot(plot_est_markov, label='Markov', kde=kde, norm_hist=True, kde_kws=kde_kws)
plt.xlabel('p(ABC)p(B) - p(AB)p(BC)')
plt.ylabel('Number of trios ABC (with order)')
plt.legend()
plt.yscale('log')

