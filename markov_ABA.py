import file_handling
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

plt.ion()
plt.close('all')
n_seeds = 6
simulations = ['f30d8a2438252005f6a9190c239c01c1']

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


def trio_prob_table(retrieved):
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode, muted_prop) = \
         file_handling.load_parameters(key)
    n_seeds = len(retrieved)

    num_ABA = np.zeros((p, p), dtype=int)
    num_AB = num_ABA.copy()
    num_B = np.zeros(p, dtype=int)

    p_B_ABA = np.nan*np.ones((p, p), dtype=float)
    p_AB_ABA = np.nan*np.ones((p, p), dtype=float)
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
                    num_AB[pattA, pattB] += 1
                    num_B[pattB] += 1
                    if sequence[ind_trans+2] == pattA:
                        num_ABA[pattA, pattB] += 1

    p_B = num_B / np.sum(num_B)
    occuring_B = num_B != 0
    p_B_ABA[:, occuring_B] = num_ABA[:, occuring_B] / num_B[occuring_B]
    occuring_AB = num_AB != 0
    p_AB_ABA[occuring_AB] = num_ABA[occuring_AB] / num_AB[occuring_AB]
    return p_B_ABA, p_AB_ABA, p_B, num_B, num_AB, num_ABA


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


alpha = 1

key = simulations[0]
retrieved = get_retrieved_seeds(key, n_seeds)
(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
 cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
 g, random_seed, p_0, n_p, nSnap, russo2008_mode, muted_prop) = \
     file_handling.load_parameters(key)
random_retrieved, shuffled_retrieved = random_eq(retrieved, n_seeds)
p_B_ABA, p_AB_ABA, p_B, num_B, num_AB, num_ABA = \
    trio_prob_table(retrieved)
p_ABA = num_ABA/np.sum(num_B)
p_AB = num_AB/np.sum(num_B)
metric = 0
metric_markov = 0
for pattA in range(p):
    for pattB in range(p):
        metric += (p_ABA[pattA, pattB]*p_B[pattB]
                   - p_AB[pattA, pattB]*p_AB[pattB, pattA])**2
print(np.sqrt(metric))
p_B_ABA_rand, p_AB_ABA_rand, p_B_rand, num_B_rand, num_AB_rand, num_ABA_rand = trio_prob_table(random_retrieved)
p_B_ABA_shuf, p_AB_ABA_shuf, p_B_shuf, num_B_shuf, num_AB_shuf, num_ABA_shuf = trio_prob_table(shuffled_retrieved)


p_B_ABA, p_AB_ABA, p_B, num_B, num_AB, num_ABA = \
    trio_prob_table(retrieved)
proba_table = np.zeros((p, p))
num_A = np.sum(num_AB, axis=1)
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

p_B_ABA_markov, p_AB_ABA_markov, p_B_markov, num_B_markov, num_AB_markov, num_ABA_markov = \
    trio_prob_table(retrieved_markov)
p_ABA_markov = num_ABA_markov/np.sum(num_B)
p_AB_markov = num_AB/np.sum(num_B)
metric = 0
metric_markov = 0
for pattA in range(p):
    for pattB in range(p):
        metric += (p_ABA_markov[pattA, pattB]*p_B_markov[pattB]
                   - p_AB_markov[pattA, pattB]*p_AB_markov[pattB, pattA])**2
print(np.sqrt(metric))
num_ABA_rand = num_ABA_rand.astype(float)+0.2
num_ABA_shuf = num_ABA_shuf.astype(float)+0.4
num_ABA_markov = num_ABA_markov.astype(float)+0.6
p_ABA = num_ABA/np.sum(num_B)
p_ABA_rand = num_ABA_rand/np.sum(num_B)
p_ABA_shuf = num_ABA_shuf/np.sum(num_B)
p_ABA_markov = num_ABA_markov/np.sum(num_B)
bins = np.arange(-0.1, 15, 0.2)
bins = bins/np.sum(num_B)
plt.close(r'p_ABA hist_g_A%.1f_a_pf%.2f' % (g_A, a_pf))
plt.figure(r'p_ABA hist_g_A%.1f_a_pf%.2f' % (g_A, a_pf))
plt.hist(np.reshape(p_ABA, p**2), alpha=alpha, bins=bins, label='Latching')
plt.hist(np.reshape(p_ABA_markov, p**2), alpha=alpha, bins=bins, label='Markov')
plt.hist(np.reshape(p_ABA_rand, p**2), alpha=alpha, bins=bins, label='Random')
plt.hist(np.reshape(p_ABA_shuf, p**2), alpha=alpha, bins=bins, label='Shuffled')
plt.legend()
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Probability of ABA transition')
plt.ylabel('Number of pair AB (with order)')
plt.title(r'$g_A$=%.1f, $a_{pf}$=%.2f' % (g_A, a_pf))

p_AB_rand = num_AB_rand/np.sum(num_B)
p_AB_shuf = num_AB_shuf/np.sum(num_B)
est = np.multiply(p_ABA, p_B[None, :]) - np.multiply(p_AB, np.transpose(p_AB))
est_rand = np.multiply(p_ABA_rand, p_B_rand[None, :]) \
    - np.multiply(p_AB_rand, np.transpose(p_AB))
est_shuf = np.multiply(p_ABA_shuf, p_B_shuf[None, :]) \
    - np.multiply(p_AB_shuf, np.transpose(p_AB_shuf))
est_markov = np.multiply(p_ABA_markov, p_B_markov[None, :]) \
    - np.multiply(p_AB_markov, np.transpose(p_AB_markov))

kde = False
plt.close('markov_test')
plt.figure('markov_test')
sns.distplot(np.reshape(est, p**2), label='Latching', kde=kde, norm_hist=True)
# sns.distplot(np.reshape(est_rand, p**2), label='Random', kde=kde)
# sns.distplot(np.reshape(est_shuf, p**2), label='Shuffled', kde=kde)
sns.distplot(np.reshape(est_markov, p**2), label='Markov', kde=kde, norm_hist=True)
plt.xlabel('p(ABA)p(B) - p(BA)p(AB)')
plt.ylabel('Number of pair AB (with order)')
plt.legend()
plt.yscale('log')


# for kick_seed in range(n_seeds):
#     print(retrieved[kick_seed][2][:10])
