##
#Find discrete maximum in a series. 
# x[i] is candidate maximum if x[i] >= x[i-1] and x[i] >= x[i+1]
# the globel maximum is selected by strategies of:
# 1. the first finding candidate maximum: suitable for a minimized n_clusters
# 2. the greatest of candidate maximum: a global maximum in the series.
def disMax(S, strategy) :
    assert type(S) == pd.Series, 'S must be pd.Series'
    strategies = {"first": lambda x: (0, x[0]),
                  "greatest": lambda x: (int(x.index[x==max(x)].values), max(x))
                  }
    cand = pd.Series()
    for i in range(1, S.size - 1) :
        if S[i] >= S[i-1] and S[i] >= S[i+1] :
            cand = cand.append(pd.Series(S[i], index=[i]))
            #cand.append((i, S[i]))
    cand = pd.Series(cand)
    return strategies[strategy](cand)

#test case
params = {"k" : [5,6,7,8,9,10,11,12,13,14,15,16,17,18]}
silhouette = []
labels     = []
for n_clusters in params["k"] :
    
    spectral = SpectralClustering(n_clusters = n_clusters,
                            eigen_solver=eigen_solver,
                            random_state=random_state,
                            # n_initial=n_initial,
                            affinity=affinity_p,
                            n_neighbors=n_neighbors)
                            # n_jobs=n_jobs)
    
    #kmeans =  KMeans(n_clusters=n_clusters, random_state=0)

    #labels = kmeans.fit_predict(affinity)
    labels0 = spectral.fit_predict(affinity) 
    labels.append(labels0 )
    
    silhouette.append(silhouette_score(dmatrix, labels0, metric='precomputed'))

result = disMax(pd.Series(silhouette), "greatest")
    
