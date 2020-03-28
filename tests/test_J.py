
class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2
rs = CustomRandomState()
rvs = stats.bernoulli(1).rvs
mask = spsp.csr_matrix(spsp.random(N, N, density=cm/N, random_state=rs, data_rvs=rvs))

print("J_i_j_k")

def J_i_j_k_l_ref(N, S, mu, a, delta, ksi_i_mu):
    J_i_j_k_l = np.zeros((N,N,S, S))
    for i in tqdm(range(N)):
        for j in range(N):
            if i != j :
                for k in range(S):
                    for l in range(S):
                        if mask.toarray()[i,j]:
                            for mu in range(p):
                                J_i_j_k_l[i,j,k,l] +=                             \
                                                (delta(ksi_i_mu[i,mu],k) - a/S)   \
                                              * (delta(ksi_i_mu[j,mu],l) - a/S)
    return J_i_j_k_l/(cm*a*(1-a/S))

def test_J(mask, delta__ksi_i_mu__k, a, S):
    test = np.tensordot((delta__ksi_i_mu__k-a/S), (delta__ksi_i_mu__k-a/S), axes=([1], [1]))
    test = tf.transpose(test, [0,2,1,3])

    mask_tf = tf.convert_to_tensor(mask.toarray(), dtype=tf.bool)
    mask_tf = tf.expand_dims(tf.cast(mask_tf, dtype=tf.double), axis=len(mask.shape))
    mask_tf = tf.expand_dims(tf.cast(mask_tf, dtype=tf.double), axis=len(mask.shape))
    
    test = mask_tf*test

    return test/(cm*a*(1-a/S))


J_i_j_k_l = J_i_j_k_l_ref(N, S, mu, a, delta, ksi_i_mu)
test = test_J(mask, delta__ksi_i_mu__k, a, S)

print()
print(np.max(np.abs(J_i_j_k_l)))
print(np.max(np.abs(test)))
print(np.log10(np.max(np.abs(J_i_j_k_l-test))))

#test = tf.convert_to_tensor(test)
#test.shape
#test.shape
##test = tf.boolean_mask(test, mask.toarray())
##test.shape

#
#test = mask_tf*test
#
#test = test.numpy()
#print()
#print(np.max(np.abs(J_i_j_k_l)))
#print(np.max(np.abs(test)))
#print(np.log10(np.max(np.abs(J_i_j_k_l-test))))