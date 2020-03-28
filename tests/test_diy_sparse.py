#J_i_j_k_l -> dense array (i,k) of sparse matrices (j,l)

data = np.zeros(len(rowInd)*S)
row = np.zeros(len(rowInd)*S, dtype ='int32')
col = row.copy()
i = 0
k = 0
for ind in range(len(rowInd)):
    j = colInd[ind]
    for l in range(S):
#        print(j,l)
        data[S*ind+l]= J_i_j_k_l[i,j,k,l]
        row[S*ind+l] = j
        col[S*ind+l] = l

A = spsp.csr_matrix((data, (row, col)), shape=(N, S))