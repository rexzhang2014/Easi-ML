# -*- coding: utf-8 -*-


#-----------------------------------------------------------------------------------
#Framework:
#1. In this framework, only set parameters then train model for you.
#2. Automatically recommend best models for you. Give you insights that what model
#    is fitting your problem best.
#3. Give you predictions if you input dataset to be predicted.
#Version: 1.0.0
#   Hard coded version. Run basic workflow of statistical modeling, including
#  One-hot preprocessing, discretize by equal-width, CART, GBDT, precision, recall
#  thresholds, PR curve with averaged precision, ROC with AUC
#Dependencies: pandas, numpy, sklearn, matplotlib
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#User Input:
#1. Dataset: the full datasets with column names.
#   1.1. Data Source: Basically from CSV file, Need support json, xml, mysql etc, for
#        mobile phone or server use.
#2. Features: a list of str , specifying the columns in the dataset
#   2.1. For better execution, it will need specification of feature data-type in next version.
#3. That's ALL!!! No need setup the models, the parameters, the evaluation metrices.
#   Forget them!
#-----------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
#from sklearn import linear_model
import matplotlib.pyplot as plt
import pipline.fitting as ft
import pipline.preprocessing as pp
from datetime import datetime

from sklearn.externals import joblib
#params: a dict of all parameters
#  minimal setting of params:
#    1. data desc: cat_features,  con_features, label
#    2. model setup: n_disc,
def EasiML_Modeling(data_ori, params) :

    frac         = params["frac"]


    data_0 = data_ori
    tic = datetime.now()
    X_norm, Y, onehot_names = pp.BuildFeatures(data_0, params)

    toc = datetime.now()

    elapsed_time = toc-tic

    print("preprocessing elapsed time: " + str(elapsed_time))

    dataset = pd.concat([X_norm, Y], axis=1)

    features= list(X_norm.columns.values)
    label   = Y.columns.values[0]
    X_train, Y_train, X_test, Y_test = ft.BuildDataSet(dataset
                        , features, label , frac )

    toc = datetime.now()
    elapsed_time = toc-tic
    print("build dataset elapsed time: " + str(elapsed_time))

    gbdt, gbdt_param = ft.GBDT(X_train, Y_train, params)
    #gbdt2=ft.GBDT(X_train, Y_train, [5, 0.5, 5, 'auto'])
    rf, rf_param = ft.RF(X_train, Y_train, params)
    #rf1 = ft.RF(X_train, Y_train, [200, 5])
    #rf2 = ft.RF(X_train, Y_train, [50, 2])
    svm, svm_param = ft.SVM(X_train, Y_train, params)

    toc = datetime.now()
    elapsed_time = toc-tic
    print("fitting elapsed time: " + str(elapsed_time))

    models = [("gbdt",gbdt),
              ("rf", rf),
              ("svm", svm)]
    roc_plt, metrics_roc = ft.eval_roc(models, X_test, Y_test)
    roc_plt.show()
    pr_plt, metrics_pr = ft.eval_pr(models, X_test, Y_test)
    pr_plt.show()

    best = ft.BestModel(metrics_roc)

    toc = datetime.now()
    elapsed_time = toc-tic
    print("total elapsed time: " + str(elapsed_time))

    joblib.dump(best[1],"bestmodel.m")
    return best , onehot_names

##data unlabeled data to be predicted
## params: at least include cat_features, con_features
def EasiML_predict(best, data, params) :

    cat_features = params["cat_features"]
    con_features = params["con_features"]
    n_disc       = params["n_disc"]
    way_disc     = params["way_disc"]
    sparse       = params["sparse"]
    dropNaN      = params["dropNaN"]

    onehot_names = params["onehot_names"]
    con_features = params["con_features"]
    cat_features = params["cat_features"]
    onehot_data  = pp.OneHotData(data[cat_features], onehot_names)
    if not sparse :
       onehot_data.columns = onehot_names.values
       if dropNaN :
          onehot_names = onehot_names.drop(onehot_names.index[onehot_names.str.contains('NULL')])
          onehot_data = onehot_data[onehot_names]

    data_onehot = onehot_data

    data_discretized = pp.DiscretizeDataFrame(data[con_features], n_disc, way_disc)

    X = pd.concat([data_onehot, data_discretized], axis=1)

    X_norm = pp.normalize(X, con_features, pp.minmax)

    m = best[1]

    result = m.predict_proba(X_norm)

    #result = pd.DataFrame(result[:, 0] > result[:, 1], columns=['pred'])

    return result
