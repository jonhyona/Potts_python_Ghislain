# ksi_i_mu = rd.randint(0,S, (N,p))
# inds = np.linspace(0,N-1,N, dtype='int')
# for mu in range(p):
#     rd.shuffle(inds)
#     ksi_i_mu[inds[:int((1-a)*N)], mu] = S
import numpy.random as rd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

rd.seed(2020)

numFact = 200
pFact = 10
S =7
p = 50
N = 200
a = 0.25
dzeta = 0.2
f = 0.25
aPf = 0.4
eps=0.000001

indChildren = np.zeros((numFact,pFact),dtype='int')
hMax = np.zeros(N)
sMax = np.zeros(N, dtype='int')
xi = S*np.ones((p,N),dtype='int')

factors = rd.randint(0,S,((N,numFact)))

for n in range(numFact):
    m = 0
    while m < pFact:
        randNum = rd.randint(0,p)
        pattPicked = False
        for i in range(m):
            if indChildren[n,i] == randNum:
                pattPicked = True
        if not pattPicked:
            indChildren[n,m] = randNum
            m +=1
            
for mu in range(p):
    hPat = np.zeros((N,S+1))
    
    for n in range(numFact):
        expon = -dzeta*n
        for m in range(pFact):
            if indChildren[n,m] == mu:
                for i in range(N):
                    y = rd.rand()/aPf
                    if y <= 1:
                        hPat[i, factors[n,i]] += y*np.exp(expon)
                        
    for i in range(N):
        randState = rd.randint(0,S)
        hPat[i,randState] += eps*rd.rand()
        
    for i in range(N):
        sMax[i] = np.argmax(hPat[i,:])
        hMax[i] = hPat[i, sMax[i]]
        
    indSorted = np.argsort(hMax)[int(N*(1-a)):]
    xi[mu, indSorted] = sMax[indSorted]

ksi_i_mu = xi.transpose()
        
    
def delta(i,j):
    return int(i==j)

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
plt.scatter(res[:,1], res[:,0], s=0.05)
plt.xlabel('C2')
plt.ylabel('C1')
plt.figure(2)
plt.hist2d(res[:,1], res[:,0], bins=20)
plt.colorbar()
plt.xlabel('C2')
plt.ylabel('C1')
        
    
        