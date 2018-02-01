# JaccardIndex 
# JI = 1 - |X ∩ Y|/|X ∪ Y|  (distance measure)
# JI = |X ∩ Y|/|X ∪ Y|  (similarity measure)
# X, Y must be same length Vectors 
def JaccardIndex(X, Y) :
    assert type(X) == pd.Series and type(Y) == pd.Series and len(X) == len(Y)
    name_X = X.loc[X==1].index
    name_Y = Y.loc[Y==1].index
    return float(np.intersect1d(name_X,name_Y).size) / float(np.union1d(name_X,name_Y).size)  


# DMatrix
# Input: 
#   df: DataFrame, each row of which indicates the observations
#   dist: Distance measure, a function f(X,Y) -> R, X,Y in Rn
# Rules: 
#   For each Oi in df, Compute DVectore between Oi and df
#   Concat Dvectors altogether
# Return: 
#   dmatrix: DataFrame of pairwise distances, shaped [n,n]
def DMatrix(df, dist) :
    assert type(df) == pd.DataFrame
    def DVector(X, y, dist) :
        dvector = X.apply(dist, args=(y,), axis=1)
        return dvector

    n = df.shape[0]
    dmatrix = pd.DataFrame([])
    for i in range(0, n) :
        dmatrix = pd.concat([dmatrix, DVector(df, df.iloc[i,:], dist)],axis=1)

    return dmatrix
    
