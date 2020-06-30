# coding=utf-8
import numpy as np
import scipy.sparse as spsp
import scipy.linalg as la
import numpy.random as rd
from parameters import N, U, S, beta, a
import file_handling
from scipy.optimize import minimize, LinearConstraint, Bounds

key = '83058b3d6f4ce563cecce654468e59ec'

delta_i_j_k_l = spsp.identity(N*(S+1), format='csc')
one_k_l = spsp.csc_matrix(np.ones((S+1, S+1)))
delta_i_j = spsp.identity(N, format='csc')

sig_vect = rd.rand(N*(S+1))

_, delta__ksi_i_mu__k, _, _ = file_handling.load_network(key)


def sig_i_l(sig_act_inact):
    mat = spsp.kron(delta_i_j, one_k_l)
    tmp = np.tile(np.reshape(sig_vect, (N*(S+1), 1)), (1, N*(S+1)))
    return mat.multiply(tmp)


def d_sig_r(sig_act_inact):
    tmp = np.tile(np.reshape(sig_vect, (N*(S+1), 1)), (1, N*(S+1)))
    return beta*(delta_i_j_k_l - sig_i_l(sig_vect)).multiply(tmp)


def k_mu__k_nu(patt1, patt2):
    k_mu = patt1/la.norm(patt1)
    tmp = patt2 - np.dot(k_mu.transpose(), patt2)
    k_nu = tmp/la.norm(tmp)
    k_mu = k_mu.tolist()
    k_nu = k_nu.tolist()
    for ii in range(N):
        k_mu.insert(ii*(S+1) + S, 0)
        k_nu.insert(ii*(S+1) + S, 0)
    return np.array(k_mu), np.array(k_nu)


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


def get_sig_act_inact(r_act_inact):
    rMax = np.max(r_act_inact)
    sig_act_inact = np.exp(beta*(r_act_inact - rMax + U_i))
    Z_i = spread_active_inactive_states.dot(
        sum_active_inactive_states.dot(sig_act_inact))
    sig_act_inact = sig_act_inact/Z_i
    return sig_act_inact


patt1 = delta__ksi_i_mu__k[:, 0] - a/S
patt2 = delta__ksi_i_mu__k[:, 1] - a/S
k_mu, k_nu = k_mu__k_nu(patt1, patt2)


# def d_m(r_act_inact):
#     sig_act_inact = get_sig_act_inact(r_act_inact)
#     right = np.dot(sig_act_inact, k_mu + k_nu)*k_mu \
#         + np.dot(sig_act_inact, k_mu)*k_nu
#     left = d_sig_r(sig_act_inact)
#     return left.dot(right)


def m_mu_nu(sig_act):
    return sig_act.dot(patt1)*sig_act.dot(patt2)


def d_m(sig_act):
    return sig_act.dot(patt1)*patt2 + sig_act.dot(patt2)*patt1


def function(sig_act): return -m_mu_nu(sig_act)


def jacobian(sig_act): return -d_m(sig_act)


def hessian(sig_act): return -2*np.outer(patt1, patt2)


sig_0 = get_sig_act_inact(rd.rand(N*(S+1)))[active]
bounds = Bounds(0, 1)
linear_constraint = LinearConstraint(sum_active_states, 0, 1)

# minimize(function, sig_0, constraints=linear_constraint,
#          bounds=bounds, jac=jacobian, hess=hessian)
