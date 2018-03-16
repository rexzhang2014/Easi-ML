def feature_eval_pairwise_scatter(ds, figuresize, outpath) :
        
    plt.figure(0,figuresize)
    
    N = ds.shape[1]
    if N > 10 : 
        N = 10
        print("Warning: N features must be fewer than 10, first 10 selected")
        
    
    for i in range(N) :
        for j in range(N) :
            plt.subplot(N,N, i * N + j + 1)
            #plt.axes((-0.1, -0.1, 1.1, 1.1))
            plt.xlim(-0.1,1.1)
            plt.ylim(-0.1,1.1)
            plt.xlabel(ds.columns.values[i])
            plt.ylabel(ds.columns.values[j])
            plt.plot(ds.iloc[:,i], ds.iloc[:,j], 'g.')
            
           
    if outpath : 
        plt.savefig(outpath)
    else : 
        plt.show()
        
    return plt

def feature_eval_correlation_heatmap(ds, method='spearman', cmap=cm.Blues,
                                     figuresize=(20,16), vmin=0, vmax=1, 
                                     outpath=None) :
    N = ds.shape[1]    
    X = np.zeros([N, N])
    for i in range(N) :
        for j in range(N) :
            X[i,j] = ds.iloc[:,i].corr(ds.iloc[:,j], method=method)
    
    plt.figure(int(np.abs(np.random.randn()*100)), figuresize)

    heatmap=plt.imshow(np.abs(X)
            ,interpolation='nearest'    
            ,cmap=cmap,aspect='auto'
            ,vmin=0,vmax=1, norm=None)
            
    plt.colorbar(mappable=heatmap,cax=None,ax=None,shrink=0.5)
    
    if outpath : 
        plt.savefig(outpath)
    else : 
        plt.show()
        
    return plt
