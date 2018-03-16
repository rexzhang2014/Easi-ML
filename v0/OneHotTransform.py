# %% OneHotName
# Input : df, a DataFrame
# Rules : Extract values of the columns as feature names, formatted as "[feature]_[value]"
def OneHotName(df) :
    assert type(df) == pd.DataFrame
    names = pd.Series()
    for col in df.columns :
        c = df[col]
        names = names.append( pd.Series(col + ':' + np.unique(c)) )
    names.index = range(0,names.shape[0])     
    return names


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
# 4. By default in pandas, pd.unique and astype('category') and fractorized(sort=True) will proceed data by lexical order. 
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
    enc = preprocessing.OneHotEncoder(categorical_features='all')
    enc.fit(df_factorized)
    onehot_data = enc.transform(df_factorized)
    if not sparse : 
       onehot_data = pd.DataFrame(onehot_data.toarray(), columns=onehot_names.values)
       if dropNaN :
          onehot_names = onehot_names.drop(onehot_names.index[onehot_names.str.contains('NULL')])
          onehot_data = onehot_data[onehot_names]
    return [onehot_data, onehot_names] if sparse else onehot_data
