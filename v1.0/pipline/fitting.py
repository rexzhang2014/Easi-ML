# -*- coding: utf-8 -*-


# Modeling Techniques:
# 1. Data Preparation: We sample from data_ori of all 1-sample and double size of 2-samples, Specifically in QDMF Product, it is 2W+ 1-samples with 4W+ 0-samples
# 2. Modeling: Train Decision Tree & GBDT for comparison. The DT is a single CART. And GBDT ensembles many CARTs in the package.
# 3. Evaluation:
#    3.1. Classification Accuracy: Accuracy on both training set and testing set.
#    3.2. Business View: DT can be visualized and for human analysis while GBDT is blackbox. Ideally DT can get more information in accordance with common sense while GBDT can provide much more accuray.
# 4. Application: Predict remaining ETB customers of QDMF purchase probability and OUTPUT top N(eg.100,500) scored client list.

###Package Importing
import numpy as np
import pandas as pd
#from sklearn import linear_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV
# Import GBDT
from sklearn.ensemble import GradientBoostingClassifier
# Import Random Forest
from sklearn.ensemble import RandomForestClassifier
# Import CART
from sklearn import tree
# Import SVC
from sklearn.svm import SVC
# Import LR
from sklearn.linear_model import LogisticRegression

###Dataset Separation###
#Input :
#  data: a DataFrame containing all data
#  X, Y: columns indicating X and Y, where Y is a size-1 list
#  frac: fractions of samples used as training set. It must fall in [0,1]
#  keys: row index if applicable. it must be as same length as data.shape[0]
#Rules :
#  Stratefied Sampling from 0/1 classes. Pick up [frac] of the total samples randomly from data
#  Get Y by X's indices, 0/1 respectively
#  Get the left as TEST set, matched by X's indices
#Return :
#  4 data sets X_train, Y_train, X_test, Y_test. Counterparty of X is in Y at the same row index
def BuildDataSet(data, X, Y, frac, keys=None) :
    if keys != None :
        data.index = data[keys]
    X_train_0 = data.loc[data[Y] == 0,X].sample(frac = frac, replace = False)
    X_train_1 = data.loc[data[Y] == 1,X].sample(frac = frac, replace = False)

    Y_train_0 = data.loc[data.index.isin(X_train_0.index), Y]
    Y_train_1 = data.loc[data.index.isin(X_train_1.index), Y]

    X_train = X_train_0.append(X_train_1)
    Y_train = Y_train_0.append(Y_train_1)


    X_test = data.loc[~data.index.isin(X_train.index),X]
    Y_test = data.loc[data.index.isin(X_test.index), Y]

    return X_train, Y_train, X_test, Y_test

# Train and evaluate
from sklearn import metrics
# Evaluate
# Input:
#   model: a fitted model implemented the sklearn interface
#   X_test, Y_test: X and Y data for test. Feature names can be omitted.
# Rules:
#   Produce precision, recall and thresholds. This function is used to find the best fitting parameters.
#   PR curve with average precision & ROC with AUC will be plotted.
# Return:
#   precision, recall, thresholds: For parameter selection use, we need know the thresholds
def Evaluate(model, X_test, Y_test) :
    pred = model.predict_proba(X_test)
    #print(pred)
    precision, recall, thresholds = metrics.precision_recall_curve(Y_test, pred[:,1])

    average_precision = metrics.average_precision_score(Y_test, pred[:,1])



    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(
              average_precision))
    plt.show()

    plt.figure().savefig("output/PR.png")

    fpr, tpr, thres_roc = metrics.roc_curve(Y_test, pred[:,1])#, pos_label=2)
    auc = metrics.roc_auc_score(Y_test, pred[:,1])

    plt.fill_between(fpr, tpr, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class ROC curve: AUC={0:0.2f}'.format(auc))
    plt.show()
    plt.figure().savefig("output/ROC.png")

    return [precision, recall, thresholds]

# Drow PR curve for a list of models
# models: list of tuples ([name], [model]). name will be shown on legends
# return the plot object
def eval_pr(models, X_test, Y_test) :
    colors = ['blue','yellow','green','red','purple','pink','grey']
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve') # AP={0:0.2f}'.format(average_precision))
    handles = []
    evals = []
    for i in range(0, len(models)) :
        m = models[i][1]
        name = models[i][0]
        pred = m.predict_proba(X_test)
        #print(pred)
        precision, recall, thresholds = metrics.precision_recall_curve(Y_test, pred[:,1])

        average_precision = metrics.average_precision_score(Y_test, pred[:,1])

        c = colors[i]
        handle, = plt.plot(recall, precision, color=c, label=name + " with ap={:0.4f}".format(average_precision))
        handles .append( handle ) #step='post', alpha=0.2, color=c)

        evals.append((name, m, [precision, recall, thresholds, average_precision]))
    plt.legend(handles=handles, loc=3)
    return plt, evals


def eval_roc(models, X_test, Y_test) :
    colors = ['blue','yellow','green','red','purple','pink','grey']
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class ROC curve') #: AUC={0:0.2f}'.format(auc))
    #plt.legend(loc)
    handles=[]
    evals=[]
    for i in range(0, len(models)) :
        m = models[i][1]
        name = models[i][0]
        pred = m.predict_proba(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(Y_test, pred[:,1])#, pos_label=2)
        auc = metrics.roc_auc_score(Y_test, pred[:,1])

        c = colors[i]
        handle, = plt.plot(fpr, tpr, color=c, label=name + " with auc={:0.4f}".format(auc))
        handles .append( handle )

        evals.append((name, m, [fpr, tpr, thresholds, auc]))

    plt.legend(handles=handles, loc=4)

    return plt, evals

##find the model whose metrics is the largest
def BestModel(metrics) :
    best = (metrics[0][0], metrics[0][1])
    for i in range(1, len(metrics)) :
        if metrics[i][2][3] > metrics[i-1][2][3] :
            best = (metrics[i][0], metrics[i][1])
    return best
## CART:
# The below functions will be organized in a factory pattern
# params will be defined for every model. 'WE' expert will define what is the major parameters for users.
def CART(X_train, Y_train, params=None, model_dump=False) :
    cart = tree.DecisionTreeClassifier()
    cart = cart.fit(X_train, Y_train)

    if model_dump :
        joblib.dump(cart,"cart.m")
    return cart
###########################################################################
##
#############################################################################
def GBDT(X_train, Y_train, model_dump=False, comprehensive=False, **kwargs) :
    sqrt_cn = int(np.floor(np.sqrt(X_train.shape[1])))
    oneten_cn = int(X_train.shape[1] / 10)
    if comprehensive :
        
        gbdt_parameters = [{'n_estimators'  : [100, 300, 600, 1000],
                            'learning_rate' : [0.05, 0.1, 0.5],
                            'max_depth'     : [1, 4, 8],
                            'max_features'  : [sqrt_cn, oneten_cn, 'auto']}]
    else : 
        gbdt_parameters = [{'n_estimators'  : [100, 300], #, 300, 600],
                            'learning_rate' : [0.05, 0.1], #, 0.5],
                            'max_depth'     : [1, 6], #, 6],
                            'max_features'  : [sqrt_cn,'auto']}]
    

    #scores = ['roc_auc']
    #score = params["score"] #'roc_auc'
    
    score = kwargs["score"]
    clf = GridSearchCV(GradientBoostingClassifier(),
                       gbdt_parameters, cv=3,
                       scoring= score ) #'%s_macro' % score)
    tic = datetime.now()
    clf.fit(X_train, Y_train)
    toc = datetime.now()

    print("GBDT Grid Search runs " + str(toc - tic))

    print("Best parameters set found on development set:")

    bestModel = clf.best_estimator_
    bestParam = clf.best_params_
    print(bestParam)
    return bestModel, bestParam

def RF(X_train, Y_train, model_dump=False, comprehensive=False, **kwargs) :
    sqrt_cn = int(np.floor(np.sqrt(X_train.shape[1])))
    oneten_cn = int(X_train.shape[1] / 10)

    if comprehensive :
                
        rf_parameters = [{'n_estimators'  : [100, 300, 600, 1000],
                            'max_depth'     : [4, 8, 16, 32],
                            'max_features'  : [sqrt_cn, oneten_cn, 'auto']}]
    else :
        rf_parameters = [{'n_estimators'  : [100, 400, 800],
                          'max_depth'     : [4, 8, 16],
                          'max_features'  : [sqrt_cn]}]
        
    #scores = ['roc_auc']
    #score = params["score"]#'roc_auc'
    score = kwargs["score"]
    clf = GridSearchCV(RandomForestClassifier(),
                       rf_parameters, cv=3,
                       scoring= score ) #'%s_macro' % score)
    tic = datetime.now()
    clf.fit(X_train, Y_train)
    toc = datetime.now()

    print("RF Grid Search runs " + str(toc - tic))



    print("Best parameters set found on development set:")

    bestModel = clf.best_estimator_
    bestParam = clf.best_params_

    print(bestParam)
    return bestModel, bestParam

#SVM
def SVM(X_train, Y_train, model_dump=False, comprehensive=False, **kwargs) :
    if comprehensive : 
        svc_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 100, 1000],
                         'max_iter':[500]},
                          {'kernel': ['linear'], 'C': [1, 100, 1000],
                           'max_iter':[500]}]
    else : 
        svc_parameters = [{'kernel': ['rbf'],
                           'gamma': [1e-3],
                           'C': [10], 
                           'max_iter':[500]},
                          {'kernel': ['linear'], 
                           'C': [10], 
                           'max_iter':[500]}]

    #scores = ['roc_auc']
    #score = params["score"]#'roc_auc'
    
    score = kwargs["score"]
    clf = GridSearchCV(SVC( probability=True),
                       svc_parameters, cv=3,
                       scoring= score ) #'%s_macro' % score)
    tic = datetime.now()
    clf.fit(X_train, Y_train)
    toc = datetime.now()

    print("SVC Grid Search runs " + str(toc - tic))

    print("Best parameters set found on development set:")

    bestModel = clf.best_estimator_
    bestParam = clf.best_params_

    print(bestParam)
    return bestModel, bestParam

#LR
def LR(X_train, Y_train, model_dump=False, comprehensive=False, **kwargs) :

    clf = LogisticRegression(tol=1e-5, random_state=0,
                             solver='liblinear', max_iter=200)
    tic = datetime.now()
    clf.fit(X_train, Y_train)
    toc = datetime.now()

    print("LR fitting runs " + str(toc - tic))

    print("Best parameters set found on development set:")

    bestModel = clf
    bestParam = clf.get_params()

    print(bestParam)
    return bestModel, bestParam


def CutOff(eval_output, metric, thred, gt = True) :
    target = eval_output.loc[eval_output[metric] >= thred,:] if gt else eval_output.loc[eval_output[metric] < thred,:]
    return target

