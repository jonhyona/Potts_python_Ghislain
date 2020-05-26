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


def network_exists(pkl_name):
    """Check if the network has already be generated"""
    return path.exists(data_path+'network_'+pkl_name)


def get_pkl_name(param=param):
    """Get the name of the pickle file corresponding to one set of
    parameters
    """
    return str(abs(hash((param)))) + '.pkl'


def get_txt_name(param=param):
    """Get the name of the text file corresponding to one set of
    parameters
    """
    return str(abs(hash(param))) + '.txt'


def save_parameters_pkl(param, pkl_name=pkl_name):
    """ Save the parameter set to a pickle file"""
    with open(data_path+'parameters_'+pkl_name, 'wb') as f:
        pickle.dump(param, f)


def save_network(ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l, pkl_name, param):
    """ Save the network to a pickle file"""
    with open(data_path+'network_'+pkl_name, 'wb') as f:
        pickle.dump((ksi_i_mu, delta__ksi_i_mu__k, J_i_j_k_l), f)


def load_network(pkl_name=pkl_name):
    """ Loads the network"""
    if not network_exists(pkl_name):
        raise 'Network not constructed : run create_pkl_file.py cue g_A tSim'
    with open(data_path+'network_'+pkl_name, 'rb') as f:
        return pickle.load(f)


def save_dynamics(cue, dynamics, txt_name=txt_name):
    """ Save dynamics obtained by run.py : transition time, overlap...."""
    save_text('dynamics_cue_%d_' % cue + txt_name, dynamics)


def load_cue_dynamics(cue, item, dtype, txt_name=txt_name):
    """Load dynamics corresponding to one particular cue and one
particular item (ether transition times, or overlaps, or
retrieved_pattern..."""
    return (load_text('dynamics_cue_%d_' % cue + txt_name)[item]).astype(dtype)


def load_full_dynamics(item, dtype, txt_name=txt_name):
    """Load dynamics from all cues"""
    res = []
    for cue in range(p):
        res.append((load_cue_dynamics(cue, item, dtype, txt_name).tolist()))
    return res


def save_parameters(pkl_name):
    with open('data_analysis/simulation_trace.csv', mode='r') as f:
        already_indexed = False
        csv_reader = csv.reader(f, delimiter=',')
        firstline = True
        for row in csv_reader:
            if firstline:
                firstline = False
            else:
                if row[1] == pkl_name:
                    already_indexed = True
                    break
    if not already_indexed:
        param = load_data('parameters_'+pkl_name)
        with open('data_analysis/simulation_trace.csv', mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([str(date.today()), pkl_name] + list(param))


def load_data(file_name, stacked=False):
    with open(data_path+file_name, 'rb') as f:
        return pickle.load(f)


def load_parameters(pkl_name):
    return load_data('parameters_'+pkl_name)


def load_retrieved(txt_name):
    return load_full_dynamics(3, int, txt_name)


def load_previously_retrieved(txt_name):
    return load_full_dynamics(4, int, txt_name)


def load_dynamics_item(item_index, pkl_name):
    item = []
    with open(data_path+'dynamics_'+pkl_name, 'rb') as f:
        while True:
            try:
                item.append(pickle.load(f)[item_index])
            except EOFError:
                break
            except pickle.UnpicklingError:
                # print(item[-1])
                break
    return item


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


def convert_seq_to_ryom_il(simulation_key):
    retrieved_saved = load_retrieved(simulation_key+'.txt')
    (dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps,
     f_russo, cm, a, U, w, tau_1, tau_2, tau_3_A,
     tau_3_B, g_A, beta, tau, t_0, g, random_seed, p_0, n_p, nSnap,
     russo2008_mode) = load_parameters(simulation_key+'.pkl')
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
