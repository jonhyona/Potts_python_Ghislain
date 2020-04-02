import scipy.sparse as spsp
from scipy import stats
import numpy as np
import numpy.random as rd
from parameters import get_parameters
from scipy.optimize import root


dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, ind_cue, \
    random_seed = get_parameters()

rd.seed(random_seed+2)


def hebbian_tensor(delta__ksi_i_mu__k):
    # class CustomRandomState(np.random.RandomState):
    #     def randint(self, k):
    #         i = rd.randint(k)   # def delta(i,j):
    #         return i - i % 2
    # rs = CustomRandomState()
    # rvs = stats.bernoulli(1).rvs

    # mask = spsp.random(N, N, density=cm/N, random_state=rs, data_rvs=rvs)
    # mask -= spsp.diags(mask.diagonal())
    # mask.eliminate_zeros()

    mask = spsp.lil_matrix((N, N))
    deck = np.linspace(0, N-1, N, dtype=int)
    for i in range(N):
        rd.shuffle(deck)
        mask[i, deck[:int(cm)]] = True
        mask[i,i] = False
    kronMask = spsp.kron(mask, np.ones((S, S)))
    kronMask = kronMask.tobsr(blocksize=(S, S))

    J_i_j_k_l = np.dot(
        (delta__ksi_i_mu__k-a/S),
        np.transpose(delta__ksi_i_mu__k-a/S))
    J_i_j_k_l = kronMask.multiply(J_i_j_k_l)/(cm*a*(1-a/S))
    
    return J_i_j_k_l.tobsr(blocksize=(S, S))


def network():
    print('Initial conditions')
    active = np.ones(N*(S+1), dtype='bool')
    inactive = active.copy()
    active[S::S+1] = False
    inactive[active] = False

    r_i_k = np.zeros(N*(S+1))
    r_i_k_act = r_i_k[active]
    r_i_S_A = g_A*r_i_k[inactive]
    r_i_S_B = (1-g_A)*r_i_k[inactive]
    # # Initializing variables
    sig_i_k = np.zeros(N*(S+1))

    m_mu = np.zeros(p)
    dt_r_i_k_act = np.zeros(r_i_k_act.shape)
    dt_r_i_S_A = np.zeros(r_i_S_A.shape)
    dt_r_i_S_B = np.zeros(r_i_S_B.shape)

    theta_i_k = np.zeros(N*S)
    dt_theta_i_k = np.zeros(theta_i_k.shape)
    h_i_k = np.zeros(theta_i_k.shape)

    # print('Find roots')
    # print('Find for A')
    fun_r_i_S_A = lambda x: g_A*S/(S+np.exp(beta*(x+U))) - x
    r_i_S_A = (root(fun_r_i_S_A, 0).x)[0]*np.ones(len(r_i_S_A))

    # print('Find for B')
    fun_r_i_S_B = lambda x: (1-g_A)*S/(S+np.exp(beta*(x+U))) - x
    r_i_S_B = (root(fun_r_i_S_B, 0).x)[0]*np.ones(len(r_i_S_B))

    theta_i_k = sig_i_k[active]
    r_i_k[active] = r_i_k_act
    r_i_k[inactive] = r_i_S_A+r_i_S_B

    return r_i_k, r_i_S_A, r_i_S_B, sig_i_k, m_mu, dt_r_i_k_act, \
        dt_r_i_S_A, dt_r_i_S_B, theta_i_k, dt_theta_i_k, h_i_k
