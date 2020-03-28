
def h_i_k_fun_ref(h_i_k, J_i_j_k_l, sig_i_k, w, S):
#    h_i_k[:,:] = np.tensordot(J_i_j_k_l, sig_i_k[:,:S], axes=([1,3], [0,1])) \
#                + w*(sig_i_k[:,:S] - 1/S*(np.sum(sig_i_k[:,:S], 1)[:, None]))
    h_i_k[:,:] *= 0
    for i in range(N):
        for k in range(S):
            h_i_k[i,k] += w*(sig_i_k[i,k])
            for l in range(S):
                h_i_k[i,k] -= w/S*sig_i_k[i,l]
                for j in range(N):
                    h_i_k[i,k] += J_i_j_k_l[i,j,k,l]*sig_i_k[j,l]
            
                
def test_fun_h(test, J_i_j_k_l, sig_i_k, w, S):
#    test[:,:]*=0
#    for ind in range(len(rowInd)):
#        i = rowInd[ind]
#        j = colInd[ind]
#        if i != j:
#            test[i,:] += np.sum(J_i_j_k_l[i,j,:,:]*sig_i_k[j,:S], 1)
    test[:,:] = np.tensordot(J_i_j_k_l, sig_i_k[:,:S], axes=([1,3], [0,1]))
    test[:,:] += w*sig_i_k[:,:S]
    test[:,:] -= w/S*np.sum(sig_i_k[:,:S, None],1)

test = r_i_k.copy()
print("    h_i_k")         
h_i_k_fun_ref(h_i_k, J_i_j_k_l, sig_i_k, w, S)
print("    sparse h-i-k")
test_fun_h(test, J_i_j_k_l, sig_i_k, w, S)

print(np.max(np.abs(h_i_k)))
print(np.max(np.abs(test)))
print(np.log10(np.max(np.abs(h_i_k-test))))

                    
#def h_i_k_fun(test, J_i_j_k_l, sig_i_k, w, S):
##    h_i_k[:,:] = np.tensordot(J_i_j_k_l, sig_i_k[:,:S], axes=([1,3], [0,1])) \
##                + w*(sig_i_k[:,:S] - 1/S*(np.sum(sig_i_k[:,:S], 1)[:, None]))
#    test[:,:] *= 0
#    for ind in range(len(rowInd)):
#        i = rowInd[ind]
#        j = colInd[ind]
#        if i != j:
#            test[i,:] = J_i_j_k_l[i,j,:,:].dot(sig_i_k[j,:S]) \
#                       + w*(sig_i_k[i,:S] - 1/S*np.sum(sig_i_k[i,:S]))
