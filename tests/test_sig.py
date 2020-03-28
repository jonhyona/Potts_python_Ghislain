# def sig_fun(sig_i_k,r_i_k, theta_i_0, U_i, beta):
#     Z_i = np.zeros(N)
#     for i in range(N):
#         for k in range(S):
#             Z_i[i] += np.exp(beta*r_i_k[i,k])
#         Z_i[i] += np.exp(beta*theta_i_0[i] + U_i[i])
#         for k in range(S):
#             sig_i_k[i,k] = np.exp(beta*r_i_k[i,k])/Z_i[i]
#         sig_i_k[i,S] =  np.exp(beta*theta_i_0[i] + U_i[i])/Z_i[i]
        
# def test_sig(test,r_i_k, theta_i_0, U_i, beta):
#     Z_i = np.zeros(N)
#     test[:,:S] = np.exp(beta*r_i_k)
#     test[:,S] = np.exp(beta*theta_i_0+ U_i)
    
#     Z_i = np.sum(test,1)
    
#     test[:,:] = test/Z_i[:,None]

        
# sig_i_k = np.zeros((N,S+1))
# test = sig_i_k.copy()

# sig_fun(sig_i_k,r_i_k, theta_i_0, U_i, beta)
# test_sig(test,r_i_k, theta_i_0, U_i, beta)

# print()
# print(np.max(np.abs(sig_i_k)))
# print(np.max(np.abs(test)))
# print(np.log10(np.max(np.abs(sig_i_k-test))))


#%%Matrix formalism
r_i_k = rd.rand(N*(S+1))
sumK = spsp.kron(spsp.eye(N), np.ones((1,S+1)))
spreadZ = spsp.kron(spsp.eye(N), np.ones((S+1,1)))

test = np.exp(beta*r_i_k)
Z_i = spreadZ.dot(sumK.dot(test))

test = test/Z_i

