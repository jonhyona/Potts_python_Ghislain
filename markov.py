import file_handling
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

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

(dt, tSim, N, S, p, num_fact, p_fact,
 dzeta, a_pf,
 eps,
 f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
 tau_3_B, g_A,
 beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
 russo2008_mode) = file_handling.load_parameters(simulations[0])


def trio_prob_table(retrieved_saved, key):
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
         file_handling.load_parameters(key)

    num_ABA = np.zeros((p, p), dtype=int)
    num_AB = num_ABA.copy()
    num_B = np.zeros(p, dtype=int)

    p_B_ABA = np.zeros((p, p), dtype=float)
    p_AB_ABA = np.zeros((p, p), dtype=float)
    p_B = np.zeros(p, dtype=float)

    for cue_ind in range(p):
        if len(retrieved_saved[cue_ind]) >= 3:
            # print(len(retrieved_saved[cue_ind]))
            duration = len(retrieved_saved[cue_ind])
            if cue_ind != retrieved_saved[cue_ind][0]:
                duration += 1
            # ind_max[cue_ind] = duration
            sequence = []
            if cue_ind != retrieved_saved[cue_ind][0]:
                sequence.append(cue_ind)
            sequence += retrieved_saved[cue_ind]
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


def random_eq(retrieved_saved):
    random_retrieved = copy.deepcopy(retrieved_saved)
    shuffled_retrieved = copy.deepcopy(retrieved_saved.copy())
    for cue_ind in range(p):
        random_retrieved[cue_ind] = list(rd.randint(0, p, len(retrieved_saved[cue_ind])))
        if retrieved_saved[cue_ind] == cue_ind:
            random_retrieved[cue_ind][0] = cue_ind
            rd.shuffle(shuffled_retrieved[cue_ind][1:])
        else:
            rd.shuffle(shuffled_retrieved[cue_ind])
    return random_retrieved, shuffled_retrieved

            
            # ind_max[cue_ind] = duration
# n_sim = len(simulations)
# # n_sim = 6
# num_ABA_plot = np.zeros(n_sim)
# num_AB_rand_plot = np.zeros(n_sim)
# num_ABA_shuf_plot = np.zeros(n_sim)
# for ii in range(0, n_sim, 3):
#     print(ii)
#     key = simulations[ii]
#     retrieved_saved = file_handling.load_retrieved(key)
#     p_B_ABA, p_AB_ABA, p_B, num_B, num_AB, num_ABA = trio_prob_table(retrieved_saved)
#     random_retrieved, shuffled_retrieved = random_eq(retrieved_saved)
#     p_B_ABA_rand, p_AB_ABA_rand, p_B_rand, num_B_rand, num_AB_rand, num_ABA_rand = trio_prob_table(random_retrieved)
#     p_B_ABA_shuf, p_AB_ABA_shuf, p_B_shuf, num_B_shuf, num_AB_shuf, num_ABA_shuf = trio_prob_table(shuffled_retrieved)

#     norm_factor = file_handling.event_counter(retrieved_saved, p)
#     num_ABA_plot[ii] = np.mean(num_ABA) / norm_factor
#     num_AB_rand_plot[ii] = np.mean(num_ABA_rand) / norm_factor
#     num_ABA_shuf_plot[ii] = np.mean(num_ABA_shuf) / norm_factor

# plt.title('Num_AB/Num_transitions')
# plt.plot(num_ABA_plot, label="Latching")
# plt.plot(num_AB_rand_plot, label='Random')
# plt.plot(num_ABA_shuf_plot, label='Shuffled')
# plt.legend()

# plt.title('Proba_ABA Knowing AB')
bins = np.arange(-0.1, 10.2, 0.2)
alpha = 1
key = simulations[1]
retrieved_saved = file_handling.load_retrieved(key)
(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
 cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
 g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
     file_handling.load_parameters(key)
random_retrieved, shuffled_retrieved = random_eq(retrieved_saved)
p_B_ABA, p_AB_ABA, p_B, num_B, num_AB, num_ABA = \
    trio_prob_table(retrieved_saved, key)
p_ABA = num_ABA/np.sum(num_B)
p_AB = num_AB/np.sum(num_B)
metric = 0
metric_markhov = 0
for pattA in range(p):
    for pattB in range(p):
        metric += (p_ABA[pattA, pattB]*p_B[pattB]
                   - p_AB[pattA, pattB]*p_AB[pattB, pattA])**2
print(np.sqrt(metric))
plt.figure(r'num_ABA hist_g_A%.1f_a_pf%.2f' % (g_A, a_pf))
p_B_ABA_rand, p_AB_ABA_rand, p_B_rand, num_B_rand, num_AB_rand, num_ABA_rand = trio_prob_table(random_retrieved, key)
p_B_ABA_shuf, p_AB_ABA_shuf, p_B_shuf, num_B_shuf, num_AB_shuf, num_ABA_shuf = trio_prob_table(shuffled_retrieved, key)
num_ABA_rand = num_ABA_rand.astype(float)+0.2
num_ABA_shuf = num_ABA_shuf.astype(float)+0.4

plt.hist(np.reshape(num_ABA, p**2), alpha=alpha, bins=bins, label='Latching')
plt.hist(np.reshape(num_ABA_rand, p**2), alpha=alpha, bins=bins, label='Random')
plt.hist(np.reshape(num_ABA_shuf, p**2), alpha=alpha, bins=bins, label='Shuffled')


p_B_ABA, p_AB_ABA, p_B, num_B, num_AB, num_ABA = \
    trio_prob_table(retrieved_saved, key)
proba_table = np.zeros((p, p))
num_A = np.sum(num_AB, axis=1)
occuring_A = num_A != 0
proba_table[occuring_A, :] = num_AB[occuring_A, :] / num_A[occuring_A, None]
retrieved_markhov = [[] for mu in range(p)]

for cue_ind in range(p):
    if len(retrieved_saved[cue_ind]) >= 3:
        # print(len(retrieved_saved[cue_ind]))
        duration = len(retrieved_saved[cue_ind])
        if cue_ind != retrieved_saved[cue_ind][0]:
            duration += 1
        retrieved_markhov[cue_ind].append(cue_ind)
    if occuring_A[cue_ind]:
            prev_mu = cue_ind
            for ind_trans in range(duration-1):
                prev_mu = rd.choice(np.array(range(p)), 1,
                                    p=proba_table[prev_mu, :].ravel())[0]
                retrieved_markhov[cue_ind].append(prev_mu)

p_B_ABA, p_AB_ABA, p_B, num_B, num_AB, num_ABA = \
    trio_prob_table(retrieved_markhov, key)
p_ABA = num_ABA/np.sum(num_B)
p_AB = num_AB/np.sum(num_B)
metric = 0
metric_markhov = 0
for pattA in range(p):
    for pattB in range(p):
        metric += (p_ABA[pattA, pattB]*p_B[pattB]
                   - p_AB[pattA, pattB]*p_AB[pattB, pattA])**2
print(np.sqrt(metric))
num_ABA = num_ABA.astype(float)+0.6
plt.hist(np.reshape(num_ABA, p**2), alpha=alpha, bins=bins, label='Markhov')
plt.legend()
plt.yscale('log')
plt.xlabel('Number of ABA transitions')
plt.ylabel('Number of pair AB (with order)')
plt.title(r'$g_A$=%.1f, $a_{pf}$=%.2f' % (g_A, a_pf))
