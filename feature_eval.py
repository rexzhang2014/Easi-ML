#plot pairwise scatter
fig = plt.figure(0,(20,16))

N = data_tmp3.shape[1]
X = np.zeros([N, N])
for i in range(N) :
    for j in range(N) :
        plt.subplot(N,N, i * N + j + 1)
        #plt.axes((-0.1, -0.1, 1.1, 1.1))
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,1.1)
        plt.xlabel(data_tmp3.columns.values[i])
        plt.ylabel(data_tmp3.columns.values[j])
        plt.plot(data_tmp3.iloc[:,i], data_tmp3.iloc[:,j], 'g.')
        
        X[i,j] = data_tmp3.iloc[:,i].corr(data_tmp3.iloc[:,j], method='spearman')

plt.savefig("fig1.png")
