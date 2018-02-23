# -*- coding: utf-8 -*-

# Preprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing 

#################################################################################
##FUNCTIONS
#################################################################################
# OneHotName
# Input : df, a DataFrame
# Rules : Extract values of the columns as feature names, formatted as "[feature]_[value]"
def OneHotName(df) :
    assert type(df) == pd.DataFrame
    names = pd.Series()
    for col in df.columns :
        c = df[col]
        for u in np.unique(c) :
            names = names.append( pd.Series(col + ':' + str(u)) )
    names.index = range(0,names.shape[0])     
    return names


def OneHotData(df, names) :
    onehot = pd.DataFrame(np.zeros([df.shape[0], names.shape[0]]), columns=names)

    for i in range(0, df.shape[0]) :
        for j in df.columns :
            onehot_col = names.index[names.where(names==j + ':' + str(df[j][i])).notnull()]
            onehot.iloc[i,onehot_col] = 1
            
    return onehot  
    
    
# Input: 
#    df: a DataFrame to be transformed
#    ID: the ID column if it is included in df; Default None
#    sparse: logically determine return a sparse matrix or a DataFrame; Default False
#    dropNaN: logic, convert NaN to 'NULL' if False, otherwise exclude it in result set
# Rules: 
# 0. assume all columns in df need be transformed and they will be changed to np.str type
# 1. Fill NaN as string "NULL" and make it as one possible value.
# 2. Do not affect ID column if applicable
# 3. Extract one-hot feature name, formatted as "[feature]_[value]"
# 4. By default in pandas, pd.unique and astype('category') will proceed data by lexical order. 
#    This ensure the correctness of the rules. 
# Return: 
#   onehot_df: a DataFrame of one-hot array if 'sparse=False';
#      otherwise a list of [sparse_matrix, columns]
def OneHotTransform(df, ID=None, sparse=False, dropNaN=False) :
    assert type(df) == pd.DataFrame
    df_data  = df.drop(ID,axis=1) if ID else df
    df_filled = df_data.fillna('NULL').apply(lambda x: x.astype(np.str),axis=0)
    onehot_names = OneHotName(df_filled)
    
    df_factorized = df_filled.apply(lambda x: x.astype('category',ordered=True).factorize(sort=True)[0],axis=0)

    onehot_dict = OneHotName(df_factorized)
    
    onehot_data = OneHotData(df_factorized, onehot_dict)
    
    if not sparse : 
       onehot_data.columns = onehot_names.values
       if dropNaN :
          onehot_names = onehot_names.drop(onehot_names.index[onehot_names.str.contains('NULL')])
          onehot_data = onehot_data[onehot_names]
    return [onehot_data, onehot_names] if sparse else onehot_data

###Preprocessing 
def minmax(x) :
    #x = np.array(x,np.float64)
    #x0 = float(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 0.0001)


def meanstd(x) :
    #print(x)
    #x = np.array(x,np.float64)
    #print(str(np.mean(x))+str(np.std(x)))
    #x1 = (x - np.mean(x)) / (np.std(x))
    #print("x1"+str(x1))
    return (x - np.mean(x)) / (np.std(x) + 0.0001)

###Normalization
def normalize(data, features, norm=None) :
    data[features] = data[features].apply(norm,axis=0)
    return data


###
#Clean the column which is actually numbers but contains comma in million and billions
###
def cleaning(x) : 
    #print(type(x.iloc[0]))
    if type(x.iloc[0]) == np.unicode or type(x.iloc[0]) == np.str :
        #print(x)
        return x.str.replace("[(), /\'\"]","")
    else :
        return x

##
#The continous feature may be loaded as string, where the number contains ',' '()' in accounting systems.
def clean_cell(cell) :
    if type(cell) == np.unicode or type(cell) == str or type(cell) == np.str :
        return cell.replace(",","")
    #elif type(cell) == np.str  :
    #    return cell.str.replace("[(), /\'\"]","")
    else :
        return cell
        
# discretize
# Input: 
#   V: the vector to be discretized
#   n: number of bins
#   way: currently support 'equal_width' only
# Rules: 
#   ways are stored in dict and call by way string
#   
# Return:
#   tagv: use the cut-off as bin tags, not simply transform the digits into 
#     integers. It can be used into 
def discretize(V, n, way) :
    def equal_width(V, n) :
        # V[V > 40000] = np.nan
        # N = V.notnull().sum()
        max_V = np.max(V)
        if max_V > 0 :
            tags = np.array(range(1, n+2)) * max_V / n
            #print(tags)
            tagv = V.apply(lambda x : tags[x * n / max_V ] if not np.isnan(x) else x)
        else :
            tagv = V.astype(np.float64)
        #print(tagv)        
        #tags = range(1,n+1) * max(V) / n
        #offset = V * n / max(V)
        #tagv = V.apply(lambda x : tags[np.floor(x * n / max(V))])
        print(tagv.index[tagv.isnull()])
        return tagv
    def equal_freq(V, n) :
        return V        
    dic = {'equal_width': equal_width, 'equal_freq': equal_freq}
    return dic[way](V, n)

#assume input df contains only continous features
def DiscretizeDataFrame(df, n, way) :
    df = df.applymap(clean_cell) \
        .fillna(0) \
        .apply(lambda x : x.astype(np.float64), axis=0) \
        .apply(discretize,  args=(n, way), axis=0)
    return df
    
#Build Features
def BuildFeatures(data, params) :
    
    cat_features = params["cat_features"]
    con_features = params["con_features"]    
    label        = params["label"]
    n_disc       = params["n_disc"]
    way_disc     = params["way_disc"]    
    sparse       = params["sparse"]
    dropNaN      = params["dropNaN"]
    
    data_onehot = OneHotTransform(data[cat_features], sparse=sparse, dropNaN=dropNaN)
    data_discretized = DiscretizeDataFrame(data[con_features], n_disc, way_disc)
    
    X = pd.concat([data_onehot, data_discretized], axis=1)
    
    X_norm = normalize(X, con_features, minmax)
    
    if label != None and label != "" :
        Y_full = OneHotTransform(data[label], sparse=sparse, dropNaN=dropNaN)
    
    Y = pd.DataFrame(Y_full.iloc[:,0])
   
    return X_norm, Y, pd.Series(data_onehot.columns)
