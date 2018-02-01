# %% OneHotName
# Input : df, a DataFrame
# Rules : Extract values of the columns as feature names, formatted as "[feature]_[value]"
def OneHotName(df) :
    assert type(df) == pd.DataFrame
    names = pd.Series()
    for col in df.columns :
        c = df[col]
        names = names.append( pd.Series(col + '_' + np.unique(c)) )
         
    return names

# %% OneHotTransform
# Input: 
#    df: a DataFrame to be transformed
#    ID: the ID column if it is included in df; Default None
#    sparse: logically determine return a sparse matrix or a DataFrame; Default False
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
def OneHotTransform(df, ID=None, sparse=False) :
    assert type(df) == pd.DataFrame
    df_data  = df.drop(ID,axis=1) if ID else df
    df_filled = df_data.fillna('NULL').apply(lambda x: x.astype(np.str),axis=0)
    onehot_names = OneHotName(df_filled)
    df_factorized = df_data.apply(lambda x: x.astype('category',ordered=True).factorize()[0],axis=0)
    enc = preprocessing.OneHotEncoder(categorical_features='all')
    enc.fit(df_factorized)
    onehot_data = enc.transform(df_factorized)
    return [onehot_data, onehot_names] if sparse else pd.DataFrame(onehot_data.toarray(), columns=onehot_names.values)
