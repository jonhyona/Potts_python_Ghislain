import numpy.random as rd
import numpy as np
from parameters import get_parameters
from parameters import get_f_russo
from time import time
import correlations
from tqdm import tqdm

dt, tSim, N, S, p, num_fact, p_fact, dzeta, a_pf, eps, cm, a, U, T, w, \
    tau_1, tau_2, tau_3_A, tau_3_B, g_A, beta, g, t_0, tau, cue_ind, \
    random_seed = get_parameters()

rd.seed(random_seed + 1)

df = 0.01
d_a_pf = 0.01

f_russo_vect = np.arange(0,1,df)
n_f = len(f_russo_vect)

a_pf_vect = np.arange(0,1,d_a_pf)
n_a_pf = len(a_pf_vect)

stored = []

filename = 'f_russo__a_pf.txt'
myfile = open(filename, 'w')

for ind_f in tqdm(range(n_f)):
    for ind_a_pf in tqdm(range(n_a_pf)):
        f_russo = f_russo_vect[ind_f]
        a_pf = a_pf_vect[ind_a_pf]
        factors = rd.binomial(1, f_russo, (N,num_fact))

        sMax = S*np.ones(N,dtype='int')
        hMax = np.zeros(N)
        ksi_i_mu = S*np.ones((N,p),dtype='int')
        ind_unit = np.linspace(0,N-1,N,dtype=int)

        gamma_mu_n = rd.rand(p, num_fact)*rd.binomial(1,a_pf,(p,num_fact))
        expo_fact = np.exp(-dzeta*np.linspace(0,num_fact-1, num_fact))

        gamma_mu_n = gamma_mu_n*expo_fact[None,:]

        sigma_n = rd.randint(0,S,(p,num_fact))

        for mu in range(p):
            fields = np.zeros((N,S))
    
            for n in range(num_fact):
                fields[:,sigma_n[mu,n]] += gamma_mu_n[mu,n]*factors[:,n]

                fields[ind_unit, rd.randint(0, S, N)[ind_unit]] += eps*rd.rand(N)

            sMax = np.argmax(fields, axis=1)
            hMax = np.max(fields, axis=1)
            indSorted = np.argsort(hMax)[int(N*(1-a)):]

            ksi_i_mu[indSorted, mu] = sMax[indSorted]

        # Compute patterns in a different form
        delta__ksi_i_mu__k = np.kron(ksi_i_mu, np.ones((S, 1)))
        k_mat = np.kron(np.ones((N, p)),
                    np.reshape(np.linspace(0, S-1, S), (S, 1)))
        delta__ksi_i_mu__k = delta__ksi_i_mu__k == k_mat

        C1C2C0 = correlations.cross_correlations(ksi_i_mu)

        x0 = np.min(C1C2C0[:, 1])
        x1 = np.max(C1C2C0[:, 1])
        y0 = np.min(C1C2C0[:, 0])
        y1 = np.max(C1C2C0[:, 0])
    
        if x0 > 0.08 and x1 < 0.6 and y1 < 0.4:
            print(f_russo, a_pf)
            myfile.write(str(f_russo) + '\n' + str(a_pf) + '\n \n')
        # else:
        #     print(x0 > 0.08, x1 < 0.6, y1 < 0.4)
myfile.close()
