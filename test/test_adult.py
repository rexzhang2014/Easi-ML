# -*- coding: utf-8 -*-
#%%
import framework as fw
import pandas as pd

#%%
data_ori = pd.read_csv("data/adult.data", low_memory=False, encoding=u'utf-8', header=None)
data_ori.columns = ["age",
"workclass",
"fnlwgt",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"capital-gain",
"capital-loss",
"hours-per-week",
"native-country",
"income"]

cat_features = ["workclass",
"education",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"native-country"]

con_features = ["age",
"fnlwgt",
"education-num",
"capital-gain",
"capital-loss",
"hours-per-week"]

label = ["income"]

params = dict()
params["con_features"] = con_features
params["cat_features"] = cat_features
params["label"]        = label
params["n_disc"]       = 5
params["frac"]         = 0.7
params["sparse"]       = False
params["dropNaN"]      = True
params["way_disc"]     = "equal_width"
#%%
model, onehot_names = fw.EasiML_Modeling(data_ori.loc[:100,:], params)

params["onehot_names"] = onehot_names

#%%
pred  = fw.EasiML_predict(model, data_ori.iloc[:10,:], params)
print(pred)
#model = fw.EasiML_Modeling(data_ori.iloc[:1000,:], params)
#pred  = fw.EasiML_predict(model, data_ori.iloc[:1000,:])
