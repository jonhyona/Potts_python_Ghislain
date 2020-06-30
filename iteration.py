"""Iterate the network

Routine listing
---------------
overlap(m_mu, delta__ksi_i_mu__k, sig_i_k)
    Overlap of the network's state with the learned patterns
h_i_k_fun(h_i_k, J_i_j_k_l, sig_i_k, delta__ksi_i_mu__k, t):
    Field to unit i state k
sig_fun(sig_i_k, r_i_k)
    Activity of units
cue(t, delta__ksi_i_mu__k)
    Additional field term to cue the network
iterate(J_i_j_k_l, delta__ksi_i_mu__k, t, analyse_time, analyse_divergence,
            sig_i_k, r_i_k, r_i_S_A, r_i_S_B, theta_i_k,
            h_i_k, m_mu, dt_r_i_S_A, dt_r_i_S_B, dt_r_i_k_act,
            dt_theta_i_k)
    Update the network

Notes
-----
All arrays are modifed in-place, which avoids to have to create and copy them
at each iteration.
"""
import numpy as np
import scipy.sparse as spsp
import time
import numpy.random as rd

from parameters import dt, N, S, a, U, tau_1, tau_2, tau_3_A, tau_3_B, p, \
    beta, g, tau, russo2008_mode

# As for now some variables contain information about active and inactive
# states, one has to be able to extract them. This shouldn't be necessary in
# future versions.
active = np.ones(N*(S+1), dtype='bool')
inactive = active.copy()
active[S::S+1] = False
inactive[active] = False

sum_active_states = spsp.kron(spsp.eye(N), np.ones((1, S)))
spread_active_states = spsp.kron(spsp.eye(N), np.ones((S, 1)))

sum_active_states = spsp.kron(spsp.eye(N), np.ones((1, S)))
spread_active_states = spsp.kron(spsp.eye(N), np.ones((S, 1)))

sum_active_inactive_states = spsp.kron(spsp.eye(N), np.ones((1, S+1)))
spread_active_inactive_states = spsp.kron(spsp.eye(N), np.ones((S+1, 1)))


U_i = U*np.zeros(N*(S+1))
U_i[S::S+1] = U*np.ones(N)


def overlap(m_mu, delta__ksi_i_mu__k, sig_i_k):
    """ Overlap of the network's state with the learned patterns"""
    m_mu[:] = 1/(a*N*(1-a/S)) \
        * np.transpose(delta__ksi_i_mu__k - a/S).dot(sig_i_k[active])


def h_i_k_fun(h_i_k, J_i_j_k_l, sig_i_k, delta__ksi_i_mu__k, t,
              cue_ind, t_0, w, cue_mask):
    """Field to unit i state k"""
    sig_i_k_act = sig_i_k[active]
    h_i_k[:] = J_i_j_k_l.dot(sig_i_k_act)
    h_i_k += w*sig_i_k_act
    if not russo2008_mode:
        h_i_k -= w/S*spread_active_states.dot(
            sum_active_states.dot(sig_i_k_act))
    h_i_k += cue(t, delta__ksi_i_mu__k, cue_ind, t_0, cue_mask)


def sig_fun(sig_i_k, r_i_k):
    """Activity of units"""
    rMax = np.max(r_i_k)
    sig_i_k[:] = np.exp(beta*(r_i_k - rMax + U_i))
    Z_i = spread_active_inactive_states.dot(
        sum_active_inactive_states.dot(sig_i_k))
    sig_i_k[:] = sig_i_k/Z_i


def get_units_to_cue(cue_ind, seed, delta__ksi_i_mu__k, muted_prop):
    """ Selects randomly units of the cued pattern that should be kicked"""
    deck = np.array(range(int(N*a)))
    gen = rd.RandomState(seed)
    gen.shuffle(deck)
    muted = deck[:int(N*a*muted_prop)]
    active = delta__ksi_i_mu__k[:, cue_ind] > 0.5
    print(active.shape)
    muted_index = np.array(range(S*N))[active]
    muted_index[muted] = 0
    cue_mask = delta__ksi_i_mu__k[:, cue_ind].copy()
    cue_mask[muted_index] = 0
    return cue_mask


def cue(t, delta__ksi_i_mu__k, cue_ind, t_0, cue_mask):
    """ Additional field term to cue the network"""
    return g * (t > t_0) \
        * np.exp(-(t-t_0)/tau) \
        * np.multiply(delta__ksi_i_mu__k[:, cue_ind], cue_mask)


def iterate(J_i_j_k_l, delta__ksi_i_mu__k, t, analyse_time, analyse_divergence,
            sig_i_k, r_i_k, r_i_S_A, r_i_S_B, theta_i_k,
            h_i_k, m_mu, dt_r_i_S_A, dt_r_i_S_B, dt_r_i_k_act,
            dt_theta_i_k, cue_ind, t_0, g_A, w, cue_mask):
    """Update the network"""
    t0 = time.time()

    sig_fun(sig_i_k, r_i_k)
    t1 = time.time()

    dt_theta_i_k[:] = (sig_i_k[active] - theta_i_k)/tau_2
    t3 = time.time()

    h_i_k_fun(h_i_k, J_i_j_k_l, sig_i_k, delta__ksi_i_mu__k, t,
              cue_ind, t_0, w, cue_mask)
    t5 = time.time()

    dt_r_i_k_act[:] = (h_i_k - theta_i_k - r_i_k[active])/tau_1
    dt_r_i_S_A[:] = (g_A*(1-sig_i_k[inactive]) - r_i_S_A)/tau_3_A
    dt_r_i_S_B[:] = ((1-g_A)*(1-sig_i_k[inactive]) - r_i_S_B)/tau_3_B
    t2 = time.time()

    r_i_k[active] += dt*dt_r_i_k_act
    r_i_S_A += dt*dt_r_i_S_A
    r_i_S_B += dt*dt_r_i_S_B
    theta_i_k += dt*dt_theta_i_k

    r_i_k[inactive] = r_i_S_A+r_i_S_B
    t6 = time.time()

    overlap(m_mu, delta__ksi_i_mu__k, sig_i_k)
    t7 = time.time()

    # Optimization and debug tool
    if analyse_time:
        print()
        print('sig update ' + str(t1-t0))
        print('r der ' + str(t2-t5))
        print('theta der update ' + str(t3-t1))
        print('h update ' + str(t5-t3))
        print('storing ' + str(t6-t2))
        print('mu update ' + str(t7-t6))
    if analyse_divergence:
        print()
        print(np.max(np.abs(h_i_k)))
        print(np.max(np.abs(dt_r_i_k_act)))
