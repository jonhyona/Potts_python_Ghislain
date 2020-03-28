def m_mu_fun_ref(m_mu,a,N,S, delta__ksi_i_mu__k, sig_i_k):
#    m_mu[:]=  np.sum(delta__ksi_i_mu__k[:,:,:] - a/S*np.ones((N, p,S))*sig_i_k[:,None,:S] \
#                /(a*N*(1-a/S)),(0,2))
    m_mu[:] *= 0
    for mu in range(p):
        for i in range(N):
            for k in range(S):
                m_mu[mu] += 1/(a*N*(1-a/S))*(delta(ksi_i_mu[i,mu],k)-a/S)*sig_i_k[i,k]

def test_fun_m(test,a,N,S, delta__ksi_i_mu__k, sig_i_k):
#    for mu in range(p):
##        for i in range(N):
###            for k in range(S):
###                test[mu] += 1/(a*N*(1-a/S))*((delta__ksi_i_mu__k[i,mu,k]-a/S)*sig_i_k[i,k])
##            test[mu] += 1/(a*N*(1-a/S))*np.sum(((delta__ksi_i_mu__k[i,mu,:]-a/S)*sig_i_k[i,:S]))
#        test[mu] += 1/(a*N*(1-a/S))*np.sum(((delta__ksi_i_mu__k[:,mu,:]-a/S)*sig_i_k[:,:S]))
    test[:] = 1/(a*N*(1-a/S))*np.sum(((delta__ksi_i_mu__k[:,:,:]-a/S)*sig_i_k[:,None,:S]),(0,2))






test = m_mu.copy()
print("m_mu")         
m_mu_fun_ref(m_mu,a,N,S, delta__ksi_i_mu__k, sig_i_k)
print("smart m_mu")
test_fun_m(test,a,N,S, delta__ksi_i_mu__k, sig_i_k)

print(np.max(np.abs(m_mu)))
print(np.max(np.abs(test)))
print(np.log10(np.max(np.abs(m_mu-test))))