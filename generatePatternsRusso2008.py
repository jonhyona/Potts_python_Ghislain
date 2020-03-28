import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
rd.seed(2020)
nFactors = 800
S=7
f = 0.2
a = 0.25
N = 200
p = 50
dzeta = 0.000001

factors = rd.binomial(1,f,(N,nFactors))

sMax = S*np.ones(N,dtype='int')
hMax = np.zeros(N)
ksi_i_mu = S*np.ones((N,p),dtype='int')

gamma_mu_n = rd.rand(p, nFactors)*rd.binomial(1,a,(p,nFactors))
expo_fact = np.exp(-dzeta*np.linspace(0,nFactors-1, nFactors))

gamma_mu_n = gamma_mu_n*expo_fact[None,:]

sigma_n = rd.randint(0,S,nFactors)

for mu in range(p):
    fields = np.zeros((N,S))

    for n in range(nFactors):
        fields[:,sigma_n[n]] += gamma_mu_n[mu,n]*factors[:,n]
    
    sMax = np.argmax(fields, axis=1)
    hMax = np.max(fields, axis=1)
    
    indSorted = np.argsort(hMax)[int(N*(1-a)):]
        
    ksi_i_mu[indSorted, mu] = sMax[indSorted]
    

def C1(ksi1, ksi2):
    return np.sum((ksi1 == ksi2)*(1-(ksi2==S)))/N/a

def C2(ksi1, ksi2):
    return np.sum((1-(ksi1==ksi2))*(1-(ksi2==S))*(1-(ksi1==S)))/N/a
    
items = [(i,j) for i in range(p) for j in range(i+1,p)]

def fun(x):
    return (C1(ksi_i_mu[:,x[0]], ksi_i_mu[:, x[1]]), C2(ksi_i_mu[:,x[0]], ksi_i_mu[:, x[1]]))

corr = map(fun, items)
res = np.array(list(corr))


plt.close('all')
plt.figure(1)
plt.subplot(121)
plt.scatter(res[:,1], res[:,0])
# plt.xlim(-0.1,0.6)
plt.xlabel('C2')
plt.ylabel('C1')
plt.subplot(122)
plt.hist2d(res[:,1], res[:,0], bins=20)
plt.colorbar()
plt.xlabel('C2')
plt.ylabel('C1')