# coding=utf-8
"""Set of functions that handle files to save or load data. They are
necessary to run the network on several cues simultaneously
"""

import csv
from datetime import date
import pickle
from os import path
from parameters import dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, \
    f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, \
    t_0, g, random_seed, p_0, n_p, nSnap, russo2008_mode
import numpy as np
import hashlib
import pandas as pd

data_path = 'data_analysis/'

# Two types of files are used. Pickles can be used to save any Python
# object, but it is less reliable than txt file. Txt file is more
# complicated to handle but offers a better control.
pkl_name = str(abs(hash((dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps,
                         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B,
                         g_A, beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
                         russo2008_mode)))) + '.pkl'
txt_name = str(abs(hash((dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps,
                         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B,
                         g_A, beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
                         russo2008_mode)))) + '.txt'
param = (dt, tSim, N, S, p, num_fact, p_fact,
         dzeta, a_pf,
         eps,
         f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
         tau_3_B, g_A,
         beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
         russo2008_mode)


def network_exists(key):
    """Check if the network has already be generated"""
    return path.exists(data_path+key+'/network.pkl')


def get_key(param):
    """Get the name of the pickle file corresponding to one set of
    parameters
    """
    return hashlib.md5(str(param).encode('utf-8')).hexdigest()


def get_txt_name(param=param):
    """Get the name of the text file corresponding to one set of
    parameters
    """
    return hashlib.md5(str(param).encode('utf-8')).hexdigest() + '.txt'


def save_parameters_pkl(param, key):
    """ Save the parameter set to a pickle file"""
    with open(data_path+key+'/parameters.pkl', 'wb') as f:
        pickle.dump(param, f)


def save_network(ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, C_i_j, key):
    """ Save the network to a pickle file"""
    with open(data_path+key+'/network.pkl', 'wb') as f:
        pickle.dump((ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, C_i_j), f)


def load_network(key):
    """ Loads the network"""
    if not network_exists(key):
        print('Network not constructed : run create_pkl_file.py cue g_A tSim')
    with open(data_path+key+'/network.pkl', 'rb') as f:
        return pickle.load(f)


def save_evolution(cue, kick_seed, m_mu_plot, key):
    save_text(key+'/evolution_cue_%d_kickseed_%d.txt' % (cue,
                                                         kick_seed),
              m_mu_plot)


def save_metrics(cue, kick_seed, d12, duration, key):
    save_text(key+'/metrics_cue_%d_kickseed_%d.txt' % (cue,
                                                       kick_seed),
              [d12,
               duration])


def save_coactivation(cue, kick_seed, coactivation, key):
    save_text(key+'/coactivation_cue_%d_kickseed_%d.txt' % (cue,
                                                            kick_seed),
              coactivation)


def load_coactivation(cue, kick_seed, key):
    return load_text(key+'/coactivation_cue_%d_kickseed_%d.txt' %
                     (cue, kick_seed))


def save_covariance(cue, kick_seed, covariance, key):
    save_text(key+'/covariance_cue_%d_kickseed_%d.txt' % (cue,
                                                          kick_seed),
              covariance)


def load_covariance(cue, kick_seed, key):
    return load_text(key+'/covariance_cue_%d_kickseed_%d.txt' % (cue,
                                                                 kick_seed))


def save_coact_pos(cue, kick_seed, coact_pos, key):
    save_text(key+'/coact_pos_cue_%d_kickseed_%d.txt' % (cue,
                                                         kick_seed),
              coact_pos)


def load_coact_pos(cue, kick_seed, key):
    return load_text(key+'/coact_pos_cue_%d_kickseed_%d.txt' % (cue,
                                                                kick_seed))


def save_coact_neg(cue, kick_seed, coact_neg, key):
    save_text(key+'/coact_neg_cue_%d_kickseed_%d.txt' % (cue,
                                                         kick_seed),
              coact_neg)


def load_coact_neg(cue, kick_seed, key):
    return load_text(key+'/coact_neg_cue_%d_kickseed_%d.txt' % (cue,
                                                                kick_seed))


def save_mean_coactivation(coactivation, key):
    save_text(key+'/coactivation_mean.txt', coactivation)


def save_tran_prop(cue, kick_seed, data, item, key):
    save_text(key+'/'+item+'_cue_%d_kickseed_%d.txt' % (cue, kick_seed), data)


def save_transition_time(cue, kick_seed, transition_time, key):
    save_tran_prop(cue, kick_seed, transition_time, 'transition_time', key)


def save_crossover(cue, kick_seed, lamb, key):
    save_tran_prop(cue, kick_seed, lamb, 'crossover', key)


def save_retrieved(cue, kick_seed, retrieved_saved, key):
    save_tran_prop(cue, kick_seed, retrieved_saved, 'retrieved', key)


def save_max_m_mu(cue, kick_seed, max_m_mu_saved, key):
    save_tran_prop(cue, kick_seed, max_m_mu_saved, 'max_m_mu', key)


def save_max2_m_mu(cue, kick_seed, max2_m_mu_saved, key):
    save_tran_prop(cue, kick_seed, max2_m_mu_saved, 'max2_m_mu', key)


def load_cue_trans_prop(cue, kick_seed, item, dtype, key):
    """Load dynamics corresponding to one particular cue and one
particular item (ether transition times, or overlaps, or
retrieved_pattern..."""
    # print(cue)
    tmp = load_text(key+'/'+item+'_cue_%d_kickseed_%d.txt' % (cue, kick_seed))
    # print(item, key)
    if len(tmp.shape) == 1:
        if (tmp.shape)[0] == 0:
            return np.array([]).astype(dtype)
    return tmp.astype(dtype)


def load_full_trans_prop(item, dtype, key, kick_seed):
    """Load dynamics from all cues"""
    res = []
    for cue in range(p):
        res.append((load_cue_trans_prop(cue, kick_seed, item, dtype,
                                        key).tolist()))
    return res


def load_mean_coactivation(coactivation, key):
    load_text(key+'/coactivation_mean.txt', coactivation)


def load_transition_time(kick_seed, key):
    return load_full_trans_prop('transition_time', float, key, kick_seed)


def load_crossover(kick_seed, key):
    return load_full_trans_prop('crossover', float, key, kick_seed)


def load_retrieved(kick_seed, key):
    return load_full_trans_prop('retrieved', int, key, kick_seed)


def load_max_m_mu(kick_seed, key):
    return load_full_trans_prop('max_m_mu', float, key, kick_seed)


def load_max2_m_mu(kick_seed, key):
    return load_full_trans_prop('max2_m_mu', float, key, kick_seed)


def load_coactivation_mean(key):
    return load_text(key+'/coactivation_mean.txt')


def save_mean_covariance(covariance, key):
    save_text(key+'/covariance_mean.txt', covariance)


def load_covariance_mean(key):
    return load_text(key+'/covariance_mean.txt')


def save_correlation_proj(cor_proj, key):
    save_text(key+'/correlation_proj.txt', cor_proj)


def load_correlation_proj(key):
    return load_text(key+'/correlation_proj.txt')


def load_metrics(key):
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, f_russo,
     cm, a, U, w, tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, tau, t_0,
     g, random_seed, p_0, n_p, nSnap, russo2008_mode) = \
        load_parameters(key)
    d12_s = np.zeros(p)
    duration_s = d12_s.copy()
    for ind_cue in range(p):
        tmp = load_text(key+'/metrics_cue_%d_kickseed_%d.txt' % ind_cue)
        d12_s[ind_cue] = tmp[0]
        duration_s[ind_cue] = tmp[1]
    return d12_s, duration_s


def load_evolution(cue, kick_seed, key):
    return load_text(key+'/evolution_cue_%d_kickseed_%d.txt' % (cue,
                                                                kick_seed))


def record_parameters(key):
    with open('data_analysis/simulation_trace.csv', mode='r') as f:
        already_indexed = False
        csv_reader = csv.reader(f, delimiter=',')
        firstline = True
        for row in csv_reader:
            if firstline:
                firstline = False
            else:
                if row[1] == key:
                    already_indexed = True
                    break
    if not already_indexed:
        param = load_parameters(key)
        with open('data_analysis/simulation_trace.csv', mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([str(date.today()), key] + list(param))


def load_data(file_name, stacked=False):
    with open(data_path+file_name, 'rb') as f:
        return pickle.load(f)


def load_parameters(key):
    return load_data(key+'/parameters.pkl')


def load_ryom_retrieved(file_name):
    res = []
    tmp = np.loadtxt(data_path+file_name, comments="#", delimiter="\n",
                     unpack=False, dtype=int)
    ind0 = 0
    for ind_event in range(len(tmp)):
        if tmp[ind_event] == -1:
            res.append(tmp[ind0:ind_event-1].tolist())
            ind0 = ind_event+1
    return res


def convert_seq_to_ryom_il(key):
    retrieved_saved = load_retrieved(key)
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps,
     f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
     tau_3_B, g_A, beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
     russo2008_mode) = load_parameters(key)
    with open(data_path+'g_A_%.2f_w_%.2f_N%d_p_%d_Runs_%.2f.txt'
              % (g_A, w, N, p, tSim), 'w') as f:
        f.write("Each line corresponds to the retrieved pattern sequence for one cue. The first pattern is the one used to cue the network. The other ones are the retrieved one.")
        for ind_cue in range(p):
            sequence = [ind_cue]
            sequence += retrieved_saved[ind_cue]
            f.write("\n")
            for ind_event in range(len(sequence)):
                f.write("%d" % sequence[ind_event])
                if ind_event < len(sequence)-1:
                    f.write(", ")


def save_text(file_name, data):
    np.savetxt(data_path+file_name, data)


def load_text(file_name):
    return(np.loadtxt(data_path + file_name))


def event_counter(retrieved, p):
    res = 0
    for cue_ind in range(p):
        res += len(retrieved[cue_ind])
    return res


def remove_duplicates(mylist):
    return list(dict.fromkeys(mylist))


def readcsv(filename):
    data = pd.read_csv(filename, header=None)
    return(np.array(data))


def get_parents_children(S, a, a_pf, p_fact, N_fact, Num_fact, zeta):
    return readcsv(
        "parents_children_S%d_a%.2f_apf%.2f_pfact%d_Nfact%d_Numfact%d_zeta%.3f"
        % (S, a, a_pf, p_fact, N_fact, Num_fact, zeta))
