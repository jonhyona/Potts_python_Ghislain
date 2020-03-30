import numpy as np
import scipy.sparse as spsp
import time

from parameters import get_parameters

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind, \
    random_seed = get_parameters()

# As one used only one dimension for units and states
active = np.ones(N*(S+1), dtype='bool')
inactive = active.copy()
active[S::S+1] = False
inactive[active] = False

sumActiveStates = spsp.kron(spsp.eye(N), np.ones((1, S)))
spreadActiveStates = spsp.kron(spsp.eye(N), np.ones((S, 1)))

sumActiveStates = spsp.kron(spsp.eye(N), np.ones((1, S)))
spreadActiveStates = spsp.kron(spsp.eye(N), np.ones((S, 1)))

sumK = spsp.kron(spsp.eye(N), np.ones((1, S+1)))
spreadZ = spsp.kron(spsp.eye(N), np.ones((S+1, 1)))

U_i = U*np.zeros(N*(S+1))
U_i[S::S+1] = U*np.ones(N)


def overlap(m_mu, delta__ksi_i_mu__k, sig_i_k):
    """ Overlap of the network's state with the learned patterns

    Parameters
    ----------
    m_mu -- 1D array
        Current overlap, that will be modifed in place
    delta__ksi_i_mu__k -- 2D array
        Stores for each pattern and unit if it is equal to k
        The first axis ranges for units and states with the convention that
        unit i state k has for index i*S+k
    sig_i_k -- 1D array
        Activation of each unit in each state
    """
    m_mu[:] = 1/(a*N*(1-a/S)) \
        * np.transpose(delta__ksi_i_mu__k - a/S).dot(sig_i_k[active])


def h_i_k_fun(h_i_k, J_i_j_k_l, sig_i_k, delta__ksi_i_mu__k, t):
    sig_i_k_act = sig_i_k[active]
    h_i_k[:] = J_i_j_k_l.dot(sig_i_k_act)
    h_i_k += w*sig_i_k_act
    h_i_k -= w/S*spreadActiveStates.dot(sumActiveStates.dot(sig_i_k_act))
    h_i_k += cue(t, delta__ksi_i_mu__k)


# def theta_i_k_der(dt_theta_i_k, sig_i_k, theta_i_k):
#     dt_theta_i_k[:] = (sig_i_k[active] - theta_i_k)/tau_2


# def r_i_k_act_der(dt_r_i_k_act, h_i_k, theta_i_k, r_i_k_act):
#     dt_r_i_k_act[:] = (h_i_k - theta_i_k - r_i_k_act)/tau_1


# def r_i_S_A_der(dt_r_i_S_A, sig_i_k, r_i_S_A):
#     dt_r_i_S_A[:] = (g_A*(1-sig_i_k[inactive]) - r_i_S_A)/tau_3_A


# def r_i_S_B_der(dt_r_i_S_B, sig_i_k, r_i_S_B):
#     dt_r_i_S_B[:] = ((1-g_A)*(1-sig_i_k[inactive]) - r_i_S_B)/tau_3_B


def sig_fun(sig_i_k, r_i_k):
    rMax = np.max(r_i_k)
    sig_i_k[:] = np.exp(beta*(r_i_k - rMax + U_i))
    Z_i = spreadZ.dot(sumK.dot(sig_i_k))
    sig_i_k[:] = sig_i_k/Z_i


def cue(t, delta__ksi_i_mu__k):
    return g * (t > t_0) \
        * 1/np.sqrt(2*np.pi*tau**2) * np.exp(-(t-t_0)**2/2/tau**2) \
        * delta__ksi_i_mu__k[:, cue_ind]


def iterate(J_i_j_k_l, delta__ksi_i_mu__k, t, analyseTime, analyseDivergence,
            sig_i_k, r_i_k, r_i_S_A, r_i_S_B, theta_i_k,
            h_i_k, m_mu, dt_r_i_S_A, dt_r_i_S_B, dt_r_i_k_act,
            dt_theta_i_k):

    t0 = time.time()
    sig_fun(sig_i_k, r_i_k)
    t1 = time.time()

    # theta_i_k_der(dt_theta_i_k, sig_i_k, theta_i_k)
    dt_theta_i_k[:] = (sig_i_k[active] - theta_i_k)/tau_2
    t3 = time.time()

    h_i_k_fun(h_i_k, J_i_j_k_l, sig_i_k, delta__ksi_i_mu__k, t)
    t5 = time.time()

    # r_i_k_act_der(dt_r_i_k_act, h_i_k, theta_i_k, r_i_k[active])
    dt_r_i_k_act[:] = (h_i_k - theta_i_k - r_i_k[active])/tau_1
    # r_i_S_A_der(dt_r_i_S_A, sig_i_k, r_i_S_A)
    dt_r_i_S_A[:] = (g_A*(1-sig_i_k[inactive]) - r_i_S_A)/tau_3_A

    # r_i_S_B_der(dt_r_i_S_B, sig_i_k, r_i_S_B)
    dt_r_i_S_B[:] = ((1-g_A)*(1-sig_i_k[inactive]) - r_i_S_B)/tau_3_B

    t2 = time.time()

    r_i_k[active] += dt*dt_r_i_k_act
    r_i_S_A += dt*dt_r_i_S_A
    r_i_S_B += dt*dt_r_i_S_B

    r_i_k[inactive] = r_i_S_A+r_i_S_B
    theta_i_k += dt*dt_theta_i_k

    t6 = time.time()
    overlap(m_mu, delta__ksi_i_mu__k, sig_i_k)
    t7 = time.time()

    if analyseTime:
        print()
        print('sig update ' + str(t1-t0))
        print('r der ' + str(t2-t5))
        print('theta der update ' + str(t3-t1))
        print('h update ' + str(t5-t3))
        print('storing ' + str(t6-t2))
        print('mu update ' + str(t7-t6))
    if analyseDivergence:
        print()
        print(np.max(np.abs(h_i_k)))
        print(np.max(np.abs(dt_r_i_k_act)))
