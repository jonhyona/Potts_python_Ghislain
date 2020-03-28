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