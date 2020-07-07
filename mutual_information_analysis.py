import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import scipy.stats as sts
# Local modules
import file_handling
import numpy.random as rd
from statistics import median

plt.ion()
plt.close('all')
n_seeds = 11

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

# simulations = ['f1691446ec79cc0cb9d1f3f898c30585',
#                'bd725ecccb665a616415eb80b3742729',
#                '779e267d7fd11b394a96bc18ac9d2261']  # w=1.4

# # simulations = ['6f276611a177a98a02697e035e772a70',
# #                '50a7e2e50bf9b00dff6cd257844d51f7',
# #                '2a123a981c3e2871ff8ff30383ecca93']  #  w=1.3

simulations_above = ['12257f9b2af7fdeaa5ebeec24b71b13c',
                     '2999e6e4eede18f9212d8abdd146e7f4',
                     '779e267d7fd11b394a96bc18ac9d2261']  # Just above the border

simulations = ['139a320d8afab59bcb18d59268071d94',
               '8d135d1215719ea60f9b7d297f1e9aeb',
               'a63c5077ffe56f59cde4dc33b7a7ac82',
               'ed63915a9b091c1018d50a3d99d48b41',
               '0786d89b52b48b48396d7f7c176fbe55',
               '9e046fe0ec70f37dc196979b42954507',
               'b4babf411ba6416e4fe2c1440a1f9fef',
               '63914c38d1d9c3b1f0f96e97fd35987b',
               '18c9dea2859f57721b6d915c267689f8',
               'f961b5cfb447cb3227dd2b5f5c71c4c1',
               'd158c4222b3dc4d3995869bccdc2f1a3',
               '89c8b67bcd8c40231b7abc89dd30a51e',
               '5e73e76d56736d4451d7b13026e9a22d',
               '72faca25053502fca9ddb571c0bdda2e',
               '7563cf938338f46c3ba9a44c6d891601'] # Correlations analysis
simulations = ['f30d8a2438252005f6a9190c239c01c1', 'eaa46e6420179b9fca55424427aa766f']
n_seeds = [11, 3]

color_s = ['blue', 'orange', 'green', 'red', 'peru', 'red', 'red', 'red', 'red', 'red']

ymin = 2**(-6)
ymax = 2**3

(dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode, muted_prop) = file_handling.load_parameters(simulations[0])
min_t = min(tau_1, tau_2, tau_3_A, tau_3_B)


def get_mi(retrieved_saved, number_limiter):
    mi_saved = []
    m_saved = []
    ent_cut = []
    ent_shifted = []
    event_counter = []
    control = []
    shuffled = []
    auto_corr_s = []
    auto_corr_shuff_s = []
    m_max = 15
    for m in range(1, m_max+1):
        # print('m = %d' % m)
        seq_cut = []
        seq_shifted = []
        auto_corr = 0
        auto_corr_shuff = 0

        for kick_seed in range(len(retrieved_saved)):
            for cue_ind in range(p):
                if len(retrieved_saved[kick_seed][cue_ind]) >= 3 + m_max:
                    # print(len(retrieved_saved[kicked_seed][cue_ind]))
                    sequence = []
                    if cue_ind != retrieved_saved[kick_seed][cue_ind][0]:
                        sequence.append(cue_ind)
                    sequence += retrieved_saved[kick_seed][cue_ind]

                    end = len(sequence)
                    ind_cut_0 = 3
                    ind_cut_1 = end - m_max
                    ind_shifted_0 = m+3
                    ind_shifted_1 = end - (m_max-m)
                    seq_cut += sequence[ind_cut_0: ind_cut_1]
                    seq_shifted += sequence[ind_shifted_0: ind_shifted_1]
                    sub_seq_cut = np.array(sequence[ind_cut_0: ind_cut_1])
                    sub_seq_shifted = np.array(sequence[ind_shifted_0:
                                                        ind_shifted_1])
                    auto_corr += np.sum(sub_seq_cut == sub_seq_shifted)

                    sub_seq_shuff = sub_seq_cut.copy()
                    rd.shuffle(sub_seq_shuff)
                    auto_corr_shuff += np.sum(sub_seq_cut == sub_seq_shuff)
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
            auto_corr_s.append(np.nan)
            auto_corr_shuff_s.append(np.nan)
        auto_corr_s.append(auto_corr)
        auto_corr_shuff_s.append(auto_corr_shuff)
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

    return m_saved, mi_saved, control, shuffled, auto_corr_s, auto_corr_shuff_s


def event_counter(retrieved):
    res = 0
    for cue_ind in range(p):
        res += len(retrieved[cue_ind])
    return res


def plot_information_flow(simulation_list):
    for ind_key in range(len(simulation_list)):
        print('ind_key = %d' % ind_key)
        simulation_key = simulation_list[ind_key]

        (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode) = file_handling.load_parameters(simulation_key)

        retrieved_saved = \
            file_handling.load_retrieved_several(n_seeds[ind_key], simulation_key)
        m_saved, mi_saved, control, shuffled = get_mi(retrieved_saved,
                                                      retrieved_saved)

        corrected = np.array(mi_saved)[:, None] - np.array(shuffled)
        # print((np.array(mi_saved)).shape)
        # print((np.array(shuffled)).shape)
        # print(corrected.shape)
        plt.title('Information flow, shuffled bias estimate')
        plt.plot(m_saved, corrected, '-o',
                 color=color_s[ind_key], label=r'$g_A$=%.1f, $w$=%.1f'
                 % (g_A, w))
        plt.plot(m_saved, shuffled, ':', color=color_s[ind_key], label='bias')
        # plt.yscale('log', basey=10)
        plt.ylim([ymin, ymax])
        plt.xlabel(r'Shift $\Delta n$')
        plt.legend(loc='upper right')


def test_shuffle_error(simulation_list, N, p):
    mi = []
    xx = np.logspace(2, np.log10(N), 100, dtype=int)
    for n in xx:
        print(np.log(n)/np.log(2))
        deck = np.arange(0, n, 1, dtype=int)
        rd.shuffle(deck)
        test_list = rd.randint(0, p, n)
        shuffled = test_list[deck]
        mi.append(mutual_information(test_list, shuffled, p)[2])

    G = 1/2/np.log(2)*(p-1)**2
    A = 0
    plt.plot(xx, mi, '--', label='Shuffled')
    plt.plot(xx, G/(A+xx), '--', label='First order Pan+96a')
    plt.yscale('log', basey=10)
    plt.xscale('log', basex=2)
    plt.xlabel('Number of samples')
    plt.ylabel('Sampling error')

    for ind_key in range(len(simulation_list)):
        print('ind_key = %d' % ind_key)
        simulation_key = simulation_list[ind_key]
        (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode) = file_handling.load_parameters(simulation_key)

        retrieved_saved = file_handling.load_retrieved(simulation_key)

        plt.vlines(event_counter(retrieved_saved), 0, 2**3, colors=color_s[ind_key])

    plt.ylim([ymin, ymax])
    plt.title('Asymptotic of shuffled estimate')
    plt.legend()
    plt.show()


def find_median(sequence):
    seq_safe = sequence.copy()
    seq_safe.sort()
    ind = len(seq_safe)//2
    print('l=%d' % len(sequence))
    return seq_safe[ind]


def get_mi_crossovers(retrieved_saved, lamb, threshold):
    m_max = 10

    mi_saved = np.zeros((m_max, m_max+1))
    m_saved = np.array(range(0, m_max+1))
    shuffled = mi_saved.copy()

    seq_cut_high = []
    seq_shifted_high = seq_cut_high.copy()
    seq_cut_low = seq_cut_high.copy()
    seq_shifted_low = seq_cut_high.copy()

    for cue_ind in range(p):
        if retrieved_saved[cue_ind] != []:
            sequence = []
            if cue_ind != retrieved_saved[cue_ind][0]:
                sequence.append(cue_ind)
            sequence += retrieved_saved[cue_ind]

            for ind_trans in range(2, len(lamb[cue_ind])-1):
                if lamb[cue_ind][ind_trans] <= threshold:
                    seq_cut_low.append(retrieved_saved[cue_ind][ind_trans])
                    seq_shifted_low.append(retrieved_saved[cue_ind][ind_trans+1])
                if lamb[cue_ind][ind_trans] >= threshold:
                    seq_cut_high.append(retrieved_saved[cue_ind][ind_trans])
                    seq_shifted_high.append(retrieved_saved[cue_ind][ind_trans+1])
                if lamb[cue_ind][ind_trans] == threshold:
                    print("bingo")

    print(len(seq_cut_high), len(seq_cut_low))
    mi_high = mutual_information(seq_cut_high, seq_shifted_high, p)[2]
    mi_low = mutual_information(seq_cut_low, seq_shifted_low, p)[2]
    seq_shuffled = seq_cut_high.copy()
    rd.shuffle(seq_shuffled)
    shuffled_high = mutual_information(seq_cut_high, seq_shuffled, p)[2]

    seq_shuffled = seq_cut_low.copy()
    rd.shuffle(seq_shuffled)
    shuffled_low = mutual_information(seq_cut_low, seq_shuffled, p)[2]

    random_high = mutual_information(seq_cut_high, rd.randint(0, p, len(seq_cut_high)), p)[2]
    random_low = mutual_information(seq_cut_low, rd.randint(0, p, len(seq_cut_low)), p)[2]

    prefactor = 0
    if len(seq_cut_high) < prefactor*p**2:
        mi_high = np.NAN
        shuffled_high = np.NAN        
    if len(seq_cut_low) < prefactor*p**2:
        mi_low = np.NAN
        shuffled_low = np.NAN

    return mi_high, mi_low, shuffled_high, shuffled_low, random_high, random_low


def compare_mi_crossover(simulation_list):
    mi_high_s = np.zeros(len(simulation_list))
    mi_low_s = mi_high_s.copy()
    shuffled_high_s = mi_high_s.copy()
    shuffled_low_s = mi_high_s.copy()
    random_low_s = mi_high_s.copy()
    random_high_s = mi_high_s.copy()
    similarity_s = mi_high.copy()
    similarity_shuf_s = mi_high.copy()
    w_s = mi_high_s.copy()
    g_A_s = mi_high_s.copy()
    threshold_s = mi_high_s.copy()

    for ind_sim in range(len(simulation_list)):
        simulation = simulation_list[ind_sim]

        retrieved_saved = file_handling.load_retrieved(simulation)
        lamb = file_handling.load_overlap(simulation)

        print('Events %d' % event_counter(lamb))

        (dt, tSim, N, S, p, num_fact, p_fact,
                 dzeta, a_pf,
                 eps,
                 f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
                 tau_3_B, g_A,
                 beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
                 russo2008_mode) = file_handling.load_parameters(simulation)

        lamb_list = [lamb[ind_cue][ind_trans] for ind_cue in
                     range(len(lamb)) for ind_trans in range(2,
                                                             len(lamb[ind_cue])-1)]
        # print(lamb_list[:10])
        threshold = median(lamb_list)
        print(threshold)

        mi_high, mi_low, shuffled_high, shuffled_low, random_high, random_low = \
            get_mi_crossovers(retrieved_saved, lamb, threshold)

        print(mi_high)

        mi_high_s[ind_sim] = mi_high
        mi_low_s[ind_sim] = mi_low
        shuffled_high_s[ind_sim] = shuffled_high
        shuffled_low_s[ind_sim] = shuffled_low
        random_low_s[ind_sim] = random_high
        random_high_s[ind_sim] = random_low
        w_s[ind_sim] = w
        g_A_s[ind_sim] = g_A
        threshold_s[ind_sim] = threshold

    ax1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=2)
    ax1.plot(g_A_s, mi_high_s-shuffled_high_s, 'b-o', label='Corrected high')
    ax1.plot(g_A_s, mi_low_s - shuffled_low_s, 'g-o', label='Corrected low')
    ax1.plot(g_A_s, mi_high_s, ':b', label='Original high')
    ax1.plot(g_A_s, mi_low_s, ':g', label='Original how')
    ax1.plot(g_A_s, shuffled_high_s, '--b', label='Bias high')
    ax1.plot(g_A_s, shuffled_low_s, '--g', label='Bias low')
    ax1.set_ylabel('Mutual information in pairs (m=2)')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_title('High-and-low-crossover mutual information')
    ax1.legend()
    ax1.set_xlabel(r'$g_A$')

    ax2 = plt.subplot2grid((3,2), (2,0))
    ax2.plot(g_A_s, w_s, '-o')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_xlabel(r'$g_A$')
    ax2.set_ylabel(r'$w$')
    ax2.set_ylim(0.9, 1.5)
    ax2.set_title('Latching border')

    ax3 = plt.subplot2grid((3,2), (2,1))
    ax3.plot(g_A_s, threshold_s, '-o')
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_xlabel(r'$g_A$')
    ax3.set_ylabel(r'$\lambda$')
    ax3.set_ylim(0, 1)
    ax3.set_title('Crossover threshold')

    plt.tight_layout()


def plot_information_flow_apf(simulation_list):
    g_A_s = np.array([0., 0.5, 1.])
    apf_s = np.array([0., 0.05, 0.1, 0.2, 0.4])
    n_gA = len(g_A_s)
    n_apf = len(apf_s)
    for ind_key in range(len(simulation_list)):
        print('ind_key = %d' % ind_key)
        simulation_key = simulation_list[ind_key]

        (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode, muted_prop) = file_handling.load_parameters(simulation_key)

        retrieved_saved = file_handling.load_retrieved_several(n_seeds[ind_key], simulation_key)
        m_saved, mi_saved, control, shuffled, auto_corr, auto_corr_shuff = \
            get_mi(retrieved_saved, retrieved_saved)
        auto_corr[1] = 0

        corrected = np.array(mi_saved)[:, None] - np.array(shuffled)
        ind_gA = [i for i in range(len(g_A_s)) if g_A_s[i] == g_A][0]
        ind_apf = [i for i in range(len(apf_s)) if apf_s[i] == a_pf][0]
        print((np.array(mi_saved)).shape)
        print((np.array(shuffled)).shape)
        print(corrected.shape)
        plt.figure('Mi')
        plt.subplot(n_gA//2 + n_gA % 2, 2, ind_gA + 1)
        plt.title('g_A=%.1f, w=%.1f' % (g_A, w))
        plt.plot(m_saved, corrected, '-o',
                 color=color_s[ind_apf], label=r'$a_{pf}$=%.2f' % a_pf)
        plt.plot(m_saved, shuffled, ':', color=color_s[ind_apf], label='bias')
        plt.yscale('log', basey=10)
        plt.ylim([ymin, ymax])
        plt.xlabel(r'Shift $\Delta n$')
        plt.legend(loc='upper right')

        plt.figure('Autocor')
        plt.subplot(n_apf//2 + n_apf % 2, 2, ind_apf + 1)
        plt.plot(m_saved, auto_corr, '-o', color=color_s[ind_gA],
                 label=r'$g_A$=%.1f' % g_A)
        plt.plot(m_saved, auto_corr_shuff, ':', color=color_s[ind_gA])
        plt.yscale('log')
        plt.title(r'$a_{pf}$=%.2f' % a_pf)
        plt.ylabel('Correlation')
        plt.xlabel(r'$\Delta n$')

    plt.figure('Mi')
    plt.tight_layout()

    plt.figure('Autocor')
    plt.legend()
    plt.tight_layout()


# plt.figure('Information flow')
# plt.subplot(121)
# plot_information_flow(simulations)
# plt.subplot(122)
# test_shuffle_error(simulations, 2**17, p)
# plt.tight_layout()

# plt.figure('Crossovers')
# compare_mi_crossover(simulations_above)

plot_information_flow_apf(simulations)
