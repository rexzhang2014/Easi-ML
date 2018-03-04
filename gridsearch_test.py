# -*- coding: utf-8 -*-
#%%
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
#from sklearn import linear_model
import matplotlib.pyplot as plt
import fitting as ft
import preprocessing as pp
from datetime import datetime
# Import GBDT
from sklearn.ensemble import GradientBoostingClassifier
# Import Random Forest
from sklearn.ensemble import RandomForestClassifier
# Import CART
from sklearn import tree
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
#%%
data_0 = data_ori.iloc[:100, :]

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
X_norm, Y, onehot_names = pp.BuildFeatures(data_0, params)

dataset = pd.concat([X_norm, Y], axis=1)

features= list(X_norm.columns.values)
label   = Y.columns.values[0]

frac    = params["frac"]
X_train, Y_train, X_test, Y_test = ft.BuildDataSet(dataset
                    , features, label , frac )

#%%
gbdt1=ft.GBDT(X_train, Y_train, [300, 0.01, 3, 'auto'])
gbdt2=ft.GBDT(X_train, Y_train, [5, 0.5, 5, 'auto'])
rf1 = ft.RF(X_train, Y_train, [200, 5])
rf2 = ft.RF(X_train, Y_train, [50, 2])

toc = datetime.now()
elapsed_time = toc-tic
print("fitting elapsed time: " + str(elapsed_time))

models = [("gbdt1",gbdt1), ("gbdt2", gbdt2),
          ("rf1", rf1), ("rf2", rf2)]
pr_plt, metrics_roc = ft.eval_roc(models, X_test, Y_test)
pr_plt.show()
pr_plt, metrics_pr = ft.eval_pr(models, X_test, Y_test)
pr_plt.show()

best = ft.BestModel(metrics_roc)

toc = datetime.now()
elapsed_time = toc-tic
print("total elapsed time: " + str(elapsed_time))
#%%
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

gbdt_parameters = [{'n_estimators'  : [20, 100],
                    'learning_rate' : [0.05, 0.1],
                    'max_depth'     : [1, 6],
                    'max_features'  : ['auto']}]
scores = ['roc_auc']
#%%

def GBDT(X_train, Y_train, params=None, model_dump=False) :
    gbdt_parameters = [{'n_estimators'  : [50, 100, 300, 600],
                        'learning_rate' : [0.05, 0.1, 0.5, 1],
                        'max_depth'     : [1, 6, 20],
                        'max_features'  : ['auto']}]
    #scores = ['roc_auc']
    score = 'roc_auc'
    clf = GridSearchCV(GradientBoostingClassifier(),
                       gbdt_parameters, cv=5,
                       scoring= score ) #'%s_macro' % score)
    clf.fit(X_train, Y_train)


    print("Best parameters set found on development set:")

    bestModel = clf.best_estimator_
    bestParam = clf.best_params_
    return bestModel, bestParam


'''
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
'''
#%%
def f(x) :
    return x, x+1

x1 = f(5)