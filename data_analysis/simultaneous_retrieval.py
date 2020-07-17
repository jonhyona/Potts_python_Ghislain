import file_handling
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.ion()
plt.close('all')

# simulations = ['a2cc92e57feefe09afa4b7d522648850']
# simulations = ['f30d8a2438252005f6a9190c239c01c1']
simulations = ['9e0fbd728bd38ee6eb130d85f35faa9a']
# simulations = ['b18e30bc89dbcb5bc2148fb9c6e0c51d']
# simulations = ['ff9fe40ed43a94577c1cc2fea6453bf0']

n_seeds = 1
key = simulations[0]

(dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo, cm,
 a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0, g,
 random_seed, p_0, n_p, nSnap, russo2008_mode, kick_prop) = \
            file_handling.load_parameters(key)

print("Loading")
retrieved = file_handling.load_retrieved_several(n_seeds, key)
crossover = file_handling.load_crossover_several(n_seeds, key)
trans_times = file_handling.load_transition_time(0,  key)

times = file_handling.load_time(0, key)
simult_ret = []

print("Analyzing")
for ind_cue in tqdm(range(1)):
    m_mu_s = file_handling.load_evolution(ind_cue, 0, key)
    tS = times[ind_cue]
    previously_retrieved = -1
    waiting_validation = False
    eta = False

    for iT in range(len(tS)):
        if tS[iT] >= trans_times[ind_cue][1]:
            m_mu = m_mu_s[iT]
            retrieved_pattern = np.argmax(m_mu)
            max_m_mu = m_mu[retrieved_pattern]
            m_mu[retrieved_pattern] = - np.inf
            outsider = np.argmax(m_mu)
            max2_m_mu = m_mu[outsider]
            m_mu[retrieved_pattern] = max_m_mu

            if retrieved_pattern != previously_retrieved \
               and not waiting_validation:
                waiting_validation = True
                print('Detected %d %d' % (retrieved_pattern, outsider))
                crossover = max_m_mu
                was_blocked = False

            if retrieved_pattern != previously_retrieved and max_m_mu > 0.5 \
               and max_m_mu - max2_m_mu > 0.2:
                print('Validated %d %d' % (retrieved_pattern, outsider))
                waiting_validation = False
                last_blocker = outsider
                last_blocked = retrieved_pattern
                previously_retrieved = retrieved_pattern
                was_blocked = False

            if waiting_validation and max_m_mu < crossover:
                crossover = max_m_mu
                trans_t = tS[iT]

            if was_blocked:
                blocked = retrieved_pattern
                blocker = outsider
                if blocker != last_blocker or blocked != last_blocked:
                    print("Blocked")
                    print(blocked, last_blocked, blocker, last_blocker)
                    simult_ret.append((last_blocked, last_blocker,
                                       ind_cue, tS[iT], iT))

            is_blocked = waiting_validation and max_m_mu > 0.5 \
                and max_m_mu - max2_m_mu <= 0.2

            if is_blocked:
                last_blocked = retrieved_pattern
                last_blocker = outsider

            was_blocked = is_blocked

dict_sim_ret = {}
for item in simult_ret:
    blocked = item[0]
    blocker = item[1]
    if not (blocked, blocker) in dict_sim_ret:
        dict_sim_ret[(blocked, blocker)] = (item[2], item[3], item[4])
    else:
        dict_sim_ret[(blocked, blocker)].append((item[2], item[3], item[4]))

cue = 0
ind_sim = 2
tS = np.array(times[cue])
recorded = tS > 0.
m_mu = np.array(file_handling.load_evolution(cue, 0, key))
blocked = simult_ret[ind_sim][0]
blocker = simult_ret[ind_sim][1]

plt.figure('visual_check_blocking_')
plt.plot(tS[recorded], m_mu[recorded, :], ':k')
plt.plot(tS[recorded], m_mu[recorded, blocked], label='blocked')
plt.plot(tS[recorded], m_mu[recorded, blocker], label='blocker')
plt.axvline(simult_ret[ind_sim][3], color='magenta')

for ind_trans in range(1, len(retrieved[0][ind_cue])):
    plt.text(trans_times[ind_cue][ind_trans], 0.8,
             str(retrieved[0][ind_cue][ind_trans]))
    plt.legend(loc='upper right')


plt.figure('behaviour_outside_blocking')
plt.plot(tS[recorded], m_mu[recorded, blocked], label='blocked')
plt.plot(tS[recorded], m_mu[recorded, blocker], label='blocker')
for ind_event in range(len(dict_sim_ret[(blocked, blocker)])):
    tt = dict_sim_ret[(blocked, blocker)][1]
    plt.axvline(tt, color='magenta')
if (blocker, blocked) in dict_sim_ret:
    for ind_event in range(len(dict_sim_ret[(blocker, blocked)])):
        tt = dict_sim_ret[(blocker, blocked)][1]
        plt.axvline(tt, color='cyan')
plt.legend()
