import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import scipy.stats as sts
# Local modules
import file_handling
import numpy.random as rd


plt.ion()
plt.close('all')

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

color_s = ['blue', 'orange', 'green']

ymin = 2**(-6)
ymax = 2**3

(dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode) = file_handling.load_parameters(simulations_above[0])
min_t = min(tau_1, tau_2, tau_3_A, tau_3_B)


def get_mi(retrieved_saved, number_limiter):
    mi_saved = []
    m_saved = []
    ent_cut = []
    ent_shifted = []
    event_counter = []
    control = []
    shuffled = []
    m_max = 10
    for m in range(1, m_max+1):
        # print('m = %d' % m)
        seq_cut = []
        seq_shifted = []

        for cue_ind in range(p):
            if len(retrieved_saved[cue_ind]) >= 3 + m_max:
                # print(len(retrieved_saved[cue_ind]))
                sequence = []
                if cue_ind != retrieved_saved[cue_ind][0]:
                    sequence.append(cue_ind)
                sequence += retrieved_saved[cue_ind]

                end = len(sequence)
                ind_cut_0 = 3
                ind_cut_1 = end - m_max
                ind_shifted_0 = m+3
                ind_shifted_1 = end - (m_max-m)
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


def plot_information_flow(simulation_list):
    for ind_key in range(len(simulations_above)):
        print('ind_key = %d' % ind_key)
        simulation_key = simulations_above[ind_key]

        (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode) = file_handling.load_parameters(simulation_key)

        retrieved_saved = file_handling.load_retrieved(simulation_key)
        m_saved, mi_saved, control, shuffled = get_mi(retrieved_saved,
                                                      retrieved_saved)

        corrected = np.array(mi_saved)[:, None] - np.array(shuffled)
        print((np.array(mi_saved)).shape)
        print((np.array(shuffled)).shape)
        print(corrected.shape)
        plt.title('Information flow, shuffled bias estimate')
        plt.plot(m_saved, corrected, '-o',
                 color=color_s[ind_key], label=r'$g_A$=%.1f, $w$=%.1f'
                 % (g_A, w))
        plt.plot(m_saved, shuffled, ':', color=color_s[ind_key], label='bias')
        plt.yscale('log', basey=2)
        plt.ylim([ymin, ymax])
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
    plt.yscale('log', basey=2)
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
                if lamb[cue_ind][ind_trans] < threshold:
                    seq_cut_low.append(retrieved_saved[cue_ind][ind_trans])
                    seq_shifted_low.append(retrieved_saved[cue_ind][ind_trans+1])
                else:
                    seq_cut_high.append(retrieved_saved[cue_ind][ind_trans])
                    seq_shifted_high.append(retrieved_saved[cue_ind][ind_trans+1])

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

    prefactor=0.1
    if len(seq_cut_high) < prefactor*p**2:
        mi_high = np.NAN
        shuffled_high = np.NAN
    if len(seq_cut_low) < prefactor*p**2:
        mi_low = np.NAN
        shuffled_low = np.NAN

    return mi_high, mi_low, shuffled_high, shuffled_low, random_high, random_low


def compare_mi_crossovere(simulation_list):
    for ind_sim in range(len(simulation_list)):
        simulation = simulation_list[ind_sim]

        retrieved_saved = file_handling.load_retrieved(simulation)
        lamb = file_handling.load_overlap(simulation)

        (dt, tSim, N, S, p, num_fact, p_fact,
                 dzeta, a_pf,
                 eps,
                 f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
                 tau_3_B, g_A,
                 beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
                 russo2008_mode) = file_handling.load_parameters(simulation)


        thresholds = np.linspace(0.1, 0.9, 10)
        mi_high_s = thresholds.copy()
        mi_low_s = thresholds.copy()
        shuffled_high_s = thresholds.copy()
        shuffled_low_s = thresholds.copy()
        random_high_s = thresholds.copy()
        random_low_s = thresholds.copy()
        for ii in range(len(thresholds)):
            threshold = thresholds[ii]
            mi_high, mi_low, shuffled_high, shuffled_low, random_high, random_low = \
                get_mi_crossovers(retrieved_saved, lamb, threshold)
            mi_high_s[ii] = mi_high
            mi_low_s[ii] = mi_low
            shuffled_high_s[ii] = shuffled_high
            shuffled_low_s[ii] = shuffled_low
            random_low_s[ii] = random_high
            random_high_s[ii] = random_low

        plt.subplot(3,2,ind_sim+1)
        plt.plot(thresholds, mi_high_s-shuffled_high_s, 'b', label='Corrected high')
        plt.plot(thresholds, mi_low_s - shuffled_low_s, 'g', label='Corrected low')
        plt.plot(thresholds, mi_high_s, ':b', label='Original high')
        plt.plot(thresholds, mi_low_s, ':g', label='Original how')
        plt.plot(thresholds, shuffled_high_s, '--b', label='Bias high')
        plt.plot(thresholds, shuffled_low_s, '--g', label='Bias low')
        plt.ylabel('Mutual information in pairs (m=2)')
        plt.xlim(0, 1)
        plt.title('w=%.2f, g_A=%.2f' % (w, g_A))
        if ind_sim == 0:
            plt.legend()
        if ind_sim in [4, 5]:
            plt.xlabel('Overlap threshold')


plt.figure('Information flow')
plt.subplot(121)
plot_information_flow(simulations_above)
plt.subplot(122)
test_shuffle_error(simulations_above, 2**17, p)
plt.tight_layout()

plt.figure('Cssovers')
