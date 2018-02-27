import pandas as pd
import numpy as np
#%%
# Series2Dataframe:
#   transform time series to data frame by the logic below: 
# Definition: 
#   Given parameter k, time series S with size N, time t, assume St ~ (St-1, St-2, ..., St-k)
#   The data frame row will consist of St-k, ..., St-2, St-1, St
#   Then any model can treat St as Y and other columns as Xts
# Properties:
#   Model can combine those Xts with any other Xp at time t to predict Y
#   The data frame shape will be (N-k, k+1). The first k elements will have no corresponding Xts
#
def Series2Dataframe(S, k ) :
    assert type(S) == pd.Series
    out = pd.DataFrame(np.zeros([N - k, k + 1]) )
    for i in range(k, S.size) :
        out.iloc[i-k, :] = pd.Series(S[i-k:i+1].values, index=range(4))
        if i % 1000 == 0 :
            print("elapsed iter : " + str(i))
    return out
