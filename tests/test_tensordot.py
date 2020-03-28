#a = np.array([[1,2],[3,4], [5,6]])
#b = np.array([[7,8], [10,11]])
#A= rd.rand(2,3)
#B = rd.rand(4,5)
#
#C = np.tensordot(A,B,axes=0)
#print(C)
#print(C.shape)

D= np.tensordot((delta__ksi_i_mu__k[:,0,:]-a/S), (delta__ksi_i_mu__k[:,0,:]-a/S), axes=0)
print(D.shape)

E = np.tensordot((delta__ksi_i_mu__k-a/S), (delta__ksi_i_mu__k-a/S), axes=([1], [1]))
print(E.shape)

E = tf.convert_to_tensor()