# =============================================================================
# First implementation of Potts Model
#   Author : Ghislain de Labbey
#   Date : 5th March 2020
#   Last update : 6th March 2020
# =============================================================================
import numpy as np
import numpy.random as rd
#import scipy.sparse.random as sprd
#rd.seed(2020)
import scipy as sp

import os
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from tqdm import tqdm

from scipy import stats
import scipy.sparse as spsp
import scipy.linalg as la
from scipy import stats
import time


rd.seed(2020)

#%%
# Simulation parameters
dt = 1
tIt = 10000
# tIt = 678


# Network parameters
# N = 1000
# S  = 5
# p = 200
# cm = 150
# a = 0.25


# tau_1 =20
# # tau_2 = 100
# tau_2 = 200
# tau_3_A = 10
# tau_3_B = 1e6
# gA = 1

# w = 1.37
# U = 0.1
# T = 0.09

N = 1000
S = 7
p = 200
a = 0.25
cm = 150
U = 0.1
T = 0.09
w = 0.8
tau_1 = 5
tau_2 = 150
tau_3_A = 2
tau_3_B = 1e6
gA = 0.3

beta = 1/T


def delta(i,j):
    return int(i==j)

# Memories
ksi_i_mu = rd.randint(0,S, (N,p))
inds = np.linspace(0,N-1,N, dtype='int')
for mu in range(p):
    rd.shuffle(inds)
    ksi_i_mu[inds[:int((1-a)*N)], mu] = S
            
delta__ksi_i_mu__k = np.zeros((N*S, p))
for i in range(N):
    for mu in range(p):
        for k in range(S):
            delta__ksi_i_mu__k[i*S+k,mu] = delta(ksi_i_mu[i,mu],k)                 

class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = rd.randint(k)
        return i - i % 2
rs = CustomRandomState()
rvs = stats.bernoulli(1).rvs

mask = spsp.random(N, N, density=cm/N, random_state=rs, data_rvs=rvs)
mask -= spsp.diags(mask.diagonal())
mask.eliminate_zeros()

rowInd = spsp.find(mask)[0]
colInd = spsp.find(mask)[1]

def test_J(mask, delta__ksi_i_mu__k, a, S):
    test = np.dot((delta__ksi_i_mu__k-a/S), np.transpose(delta__ksi_i_mu__k-a/S))
    
    kronMask = spsp.kron(mask, np.ones((S,S)))
    
    test = kronMask.multiply(test)/(cm*a*(1-a/S))
    return test.tobsr(blocksize=(S,S))

J_i_j_k_l = test_J(mask, delta__ksi_i_mu__k, a, S)


active = np.ones(N*(S+1), dtype='bool')
inactive = active.copy()
active[S::S+1] = False

inactive[active] = False
sumActiveStates = spsp.kron(spsp.eye(N), np.ones((1,S)))
spreadActiveStates = spsp.kron(spsp.eye(N), np.ones((S,1)))

def test_fun_h(test, J_i_j_k_l, sig_i_k, w, S):
    th1 = time.time()
    sig_i_k_act = sig_i_k[active]
    test[:] = J_i_j_k_l.dot(sig_i_k_act)
    th2 = time.time()
    test += w*sig_i_k_act
    th3 = time.time()
    test -= w/S*spreadActiveStates.dot(sumActiveStates.dot(sig_i_k_act))
    th4 = time.time()
    # print('h')
    # print(th2-th1)
    # print(th3-th2)
    # print(th4-th3)
def h_i_k_fun(h_i_k, J_i_j_k_l, sig_i_k, w, S):
    return test_fun_h(h_i_k, J_i_j_k_l, sig_i_k, w, S)

def theta_i_k_der(dt_theta_i_k, sig_i_k, theta_i_k, tau_2):
    dt_theta_i_k[:] = (sig_i_k[active] - theta_i_k)/tau_2
            
def r_i_k_act_der(dt_r_i_k_act,h_i_k, theta_i_k, r_i_k, tau_1):
    dt_r_i_k_act[:] = (h_i_k - theta_i_k - r_i_k_act)/tau_1
    
def r_i_S_A_der():
    dt_r_i_S_A[:] = (gA*(1-sig_i_k[inactive]) - r_i_S_A)/tau_3_A
def r_i_S_B_der():
    dt_r_i_S_B[:] = ((1-gA)*(1-sig_i_k[inactive]) - r_i_S_B)/tau_3_B
                 
sumK = spsp.kron(spsp.eye(N), np.ones((1,S+1)))
spreadZ = spsp.kron(spsp.eye(N), np.ones((S+1,1)))

def test_sig(test,r_i_k, beta):
    rMax = np.max(r_i_k)
    test[:] = np.exp(beta*(r_i_k - rMax + U_i))
    Z_i = spreadZ.dot(sumK.dot(test))  
    test[:] = test/Z_i
def sig_fun(sig_i_k,r_i_k, beta):
    test_sig(sig_i_k,r_i_k, beta)

def test_fun_m(test,a,N,S, delta__ksi_i_mu__k, sig_i_k):
    test[:] = 1/(a*N*(1-a/S))*np.transpose(delta__ksi_i_mu__k-a/S).dot(sig_i_k[active])
    
def m_mu_fun(m_mu,a,N,S, delta__ksi_i_mu__k, sig_i_k):
    return test_fun_m(m_mu ,a,N,S, delta__ksi_i_mu__k, sig_i_k)
    

# #%% Integration
                
# Initial condition
print('Initial conditions')
r_i_k = np.zeros(N*(S+1))

for i in range(N):
    for k in range(S):
        r_i_k[i*(S+1)+k] = delta(ksi_i_mu[i,0], k)
r_i_k_act = r_i_k[active]
r_i_S_A = gA*r_i_k[inactive]
r_i_S_B = (1-gA)*r_i_k[inactive]

# # Initializing variables
sig_i_k = np.zeros(N*(S+1))
theta_i_0 = np.zeros(N)
U_i = U*np.zeros(N*(S+1))
U_i[S::S+1] = U*np.ones(N)

m_mu = np.zeros(p)
dt_r_i_k_act = np.zeros(r_i_k_act.shape)
dt_r_i_S_A = np.zeros(r_i_S_A.shape)
dt_r_i_S_B = np.zeros(r_i_S_B.shape)

theta_i_k = np.zeros(N*S)
dt_theta_i_k = np.zeros(theta_i_k.shape)
h_i_k = np.zeros(theta_i_k.shape)
            
print('Intégration')
tS = np.arange(0,tIt,dt)
nT = tS.shape[0]

# Plot parameters
nSnap = nT
r_i_k_plot = np.zeros((nSnap,N*(S+1)))
m_mu_plot = np.zeros((nSnap, p))
theta_i_k_plot = np.zeros((nSnap, N*S))
sig_i_k_plot = np.zeros((nSnap, N*(S+1)))

analyseTime = False
analyseDivergence = False
iT = 0

#%%
for iT in tqdm(range(nT)):
# for iT in range(nT):
    t0 = time.time()
    sig_fun(sig_i_k,r_i_k, beta)
    t1 = time.time()
    
    theta_i_k_der(dt_theta_i_k, sig_i_k, theta_i_k, tau_2)
    t3 = time.time()

    h_i_k_fun(h_i_k, J_i_j_k_l, sig_i_k, w, S)
    t5 = time.time()
    
    r_i_k_act_der(dt_r_i_k_act,h_i_k, theta_i_k, r_i_k, tau_1)
    r_i_S_A_der()
    r_i_S_A_der()
    t2 = time.time()
    
    r_i_k_act += dt*dt_r_i_k_act
    r_i_S_A += dt*dt_r_i_S_A
    r_i_S_B += dt*dt_r_i_S_B
    r_i_k[active] = r_i_k_act
    r_i_k[inactive] = r_i_S_A+r_i_S_B
    
    theta_i_k += dt*dt_theta_i_k

    
    t6 = time.time()
    m_mu_fun(m_mu,a,N,S, delta__ksi_i_mu__k, sig_i_k)
    t7 = time.time()

    
    r_i_k_plot[iT,:] = r_i_k
    m_mu_plot[iT,:] = m_mu
    theta_i_k_plot[iT,:] = theta_i_k
    sig_i_k_plot[iT,:] = sig_i_k
    t8 = time.time()
    
    if analyseTime:
        print()
        print(iT)
        print('sig update ' + str(t1-t0))
        print('r der ' + str(t2-t5))
        print('theta der update ' + str(t3-t1))
        print('h update ' + str(t5-t3))
        print('storing ' +str(t6-t2))
        print('mu update ' + str(t7-t6))
        print('save ' + str(t8-t7))
    if analyseDivergence:
        print()
        print(iT)
        print(np.max(np.abs(h_i_k)))
        print(np.max(np.abs(dt_r_i_k)))

# %%Plot
plt.close('all')

# plt.figure(1)

# ax1 = plt.subplot(3,2,1)
# for i in range(0,N,N//10):
#     for k in range(S+1):
#         ax1.plot(tS[:], r_i_k_plot[:,i*(S+1)+k])
        
# ax2 = plt.subplot(3,2,2)
# for i in range(0,N,N//10):
#     for k in range(S):
#         ax2.plot(tS[:], theta_i_k_plot[:, i*S+ k])
        
# ax3 = plt.subplot(3,2,3)
# for mu in range(p):
#     ax3.plot(tS, m_mu_plot[:, mu])
    

# ax5 = plt.subplot(3,2,5)
# for i in range(0,N,N//10):
#     for k in range(S):
#         ax5.plot(tS, sig_i_k_plot[:,i*(S+1) +k])
    
# ax6 = plt.subplot(3,2,4)
# for i in range(0,N,N//10):
#     for k in range(S):
#         ax6.plot(tS, sig_i_k_plot[:,i*(S+1) + S])


# ax1.set_title("r")

# ax2.set_title("theta_k")

# ax3.set_title("overlap")

# ax5.set_title("sig_i_k")

# ax6.plot("sig_i_S")
plt.figure(2)
for mu in range(p):
    plt.plot(tS, m_mu_plot[:, mu])
plt.title('overlap')

plt.show()