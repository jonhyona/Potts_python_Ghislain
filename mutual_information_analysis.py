import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import scipy.stats as sts
# Local modules
import file_handling
import numpy.random as rd


def mutual_information(var1_realisations, var2_realisations, p):
    bins1 = np.arange(-0.5, p+.5, 1)
    bins2 = np.arange(-0.5, p+0.5, 1)
    hist1 = np.histogram(var1_realisations, bins=bins1, density=True)[0]
    hist2 = np.histogram(var2_realisations, bins=bins2, density=True)[0]
    cross_hist = np.histogram2d(var1_realisations, var2_realisations, bins=(bins1, bins2), normed=True)[0]
    cross_hist = np.reshape(cross_hist, (cross_hist.shape[0]*cross_hist.shape[1], 1))
    ent_cut = sts.entropy(hist1, base=2)
    ent_shifted = sts.entropy(hist2, base=2)
    ent_joint = ent_cut + ent_shifted - sts.entropy(cross_hist, base=2)
    return ent_cut, ent_shifted, ent_joint


simulations = ['134332572273764016',
               '5630274950760470864',
               '2756919133842347063']
ryom_data = ['seq_w1.4_gA0.0', 'seq_w1.4_gA0.5', 'seq_w1.4_gA1.0']
color_s = ['blue', 'orange', 'green']
color_s_ryom = ['navy', 'peru', 'darkolivegreen']


(dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode) = file_handling.load_parameters(simulations[0]+'.pkl')
min_t = min(tau_1, tau_2, tau_3_A, tau_3_B)


def get_mi(retrieved_saved):
    mi_saved = []
    m_saved = []
    ent_cut = []
    ent_shifted = []
    event_counter = []
    control = []
    shuffled = []
    m_max = 10
    for m in range(1, m_max+1):
        print('m = %d' % m)
        seq_cut = []
        seq_shifted = []
        for cue_ind in range(p):
            sequence = [cue_ind]
            sequence += retrieved_saved[cue_ind]
            ind_cut_0 = 3
            ind_cut_1 = len(sequence) - m_max
            ind_shifted_0 = m+3
            ind_shifted_1 = len(sequence) - (m_max-m)
            seq_cut += sequence[ind_cut_0: ind_cut_1]
            seq_shifted += sequence[ind_shifted_0: ind_shifted_1]
        if seq_cut == []:
            break
        seq_cut = np.array(seq_cut)
        seq_shifted = np.array(seq_shifted)
        deck = np.arange(0, len(seq_cut), 1, dtype=int)
        mi = mutual_information(seq_cut, seq_shifted, p)
        if m == 1:
            mi_saved.append(mi[0])
            m_saved.append(0)
            randomized = rd.randint(0, p, len(seq_cut))
            rd.shuffle(deck)
            shuffled_seq = seq_cut[deck]
            control.append(mutual_information(seq_cut, randomized, p)[2])
            shuffled.append(mutual_information(seq_cut, shuffled_seq, p)[2])
        mi_saved.append(mi[2])
        ent_cut.append(mi[0])
        ent_shifted.append(mi[1])
        m_saved.append(m)
        event_counter.append(len(seq_cut))
        randomized = rd.randint(0, p, len(seq_cut))
        control.append(mutual_information(seq_cut, randomized, p)[2])
        rd.shuffle(deck)
        shuffled_seq = seq_cut[deck]
        shuffled.append(mutual_information(seq_cut, shuffled_seq, p)[2])

    return m_saved, mi_saved, control, shuffled


def event_counter(retrieved):
    res = 0
    for cue_ind in range(p):
        res += len(retrieved[cue_ind])
    return res

for ind_key in range(len(simulations)):
    print('ind_key = %d' % ind_key)
    simulation_key = simulations[ind_key]
    ryom_name = ryom_data[ind_key]
    (dt, tSim, N, S, p, num_fact, p_fact,
     dzeta, a_pf,
     eps,
     f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
     tau_3_B, g_A,
     beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
     russo2008_mode) = file_handling.load_parameters(simulation_key+'.pkl')

    retrieved_saved = file_handling.load_retrieved(simulation_key+'.txt')
    ryom_retrieved = file_handling.load_ryom_retrieved(ryom_name)

    m_saved, mi_ryom, control_ryom, shuffled_ryom = get_mi(ryom_retrieved)
    m_saved, mi_saved, control, shuffled = get_mi(retrieved_saved)

    plt.subplot(221)
    plt.title('Gsln and Ryom data with shuffled control')
    plt.plot(m_saved, mi_saved, '-o',color=color_s[ind_key], label='Gsln, g_A='+str(g_A))
    plt.plot(m_saved, shuffled, ':', color=color_s[ind_key])
    plt.plot(m_saved, mi_ryom, '-o',color=color_s_ryom[ind_key], label='Ryom, g_A='+str(g_A))
    plt.plot(m_saved, shuffled_ryom, ':', color=color_s_ryom[ind_key])
    plt.yscale('log', basey=2)
    plt.legend()

    plt.subplot(222)
    plt.title('Gsln data, with bias estimates')
    plt.plot(m_saved, mi_saved, '-o',color=color_s[ind_key])
    plt.plot(m_saved, shuffled, ':', color=color_s[ind_key], label='Shuffled')
    plt.hlines(1/2/event_counter(retrieved_saved)/np.log(2)*(p-1)**2, 0, np.max(m_saved), linestyle='dashed', colors=color_s[ind_key], label='First order Pan+96a')
    plt.yscale('log', basey=2)
    plt.legend()


    plt.subplot(223)
    plt.title('Ryom data, with bias estimates')
    plt.plot(m_saved, mi_ryom, '-o',color=color_s_ryom[ind_key])
    plt.plot(m_saved, shuffled_ryom, ':', color=color_s_ryom[ind_key], label='Shuffled')
    plt.hlines(1/2/event_counter(ryom_retrieved)/np.log(2)*(p-1)**2, 0, np.max(m_saved), linestyle='dashed', colors=color_s_ryom[ind_key], label='First order Pan+96a')
    plt.yscale('log', basey=2)
    plt.legend()

plt.subplot(224)
plt.title('Mutual information between random [1, p] list and shuffled, p = %d' % p)

def test_shuffle_error(N, p):
    mi = []
    xx = np.logspace(2, np.log10(N), 100, dtype=int)
    for n in xx:
        print(np.log(n)/np.log(2))
        deck = np.arange(0, n, 1, dtype=int)
        rd.shuffle(deck)
        test_list = rd.randint(0, p, n)
        shuffled = test_list[deck]
        mi.append(mutual_information(test_list, shuffled, p)[2])

    plt.plot(xx, mi, '--', label='Shuffled')
    plt.plot(xx, 1/2/xx/np.log(2)*(p-1)**2, '--', label='First order Pan+96a')
    plt.yscale('log', basey=2)
    plt.xscale('log', basex=2)
    plt.xlabel('Number of samples')
    plt.ylabel('Sampling error')

    for ind_key in range(len(simulations)):
        print('ind_key = %d' % ind_key)
        simulation_key = simulations[ind_key]
        ryom_name = ryom_data[ind_key]
        (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode) = file_handling.load_parameters(simulation_key+'.pkl')

        retrieved_saved = file_handling.load_retrieved(simulation_key+'.txt')
        ryom_retrieved = file_handling.load_ryom_retrieved(ryom_name)

        plt.vlines(event_counter(retrieved_saved), 0, 2**3, colors=color_s[ind_key])
        plt.vlines(event_counter(ryom_retrieved), 0, 2**3, colors=color_s_ryom[ind_key])

    plt.legend()
    plt.show()

test_shuffle_error(2**17, p)
        # p1 = 0.1
# p2 = 0.2
# X1 = rd.binomial(1, p1, 10000)
# Y1 = X1+rd.binomial(1, p1, 10000)
# ent1 = -(p1*np.log(p1) + (1-p1)*np.log(1-p1))
# ent2 = -((1-p1)**2*np.log((1-p1)**2) + 2*p1*(1-p1)*np.log(2*p1*(1-p1)) +p1**2*np.log(p1**2))
# joint_ent = -((1-p1)**2*np.log((1-p1)**2) + 2*p1*(1-p1)*np.log(p1*(1-p1)) + p1**2*np.log(p1**2))
# print(ent1, ent2, ent1+ent2-joint_ent)
# print(mutual_information(X1, Y1))
