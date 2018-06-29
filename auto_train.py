
###Package Importing 
import numpy as np
import pandas as pd

from sklearn import metrics

import logging 
from sklearn.externals import joblib
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

from datetime import datetime

from preprocessing import hash_col

from preprocessing import onehot

from preprocessing import normalize
        
from model import machine_learning as ml

from preprocessing import discretize
#from sklearn.ensemble import RandomForestClassifier
wkdir = ""

#set parameters 
dump_path = wkdir + "/" + "model"
data_path = wkdir + "/" + "data"
out_path  = wkdir + "/" + "output"

colList_float = []

colList_cnt = []

colList_days = []

colList_unicode = []

colList_dup = []


keys = ''
labels = ''
 
def train(*args, **kwargs) :
    
    ###Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info("START training")
    
    # Mandatory Args
    wkdir, dump_path, data_path, out_path = kwargs["path_list"]
   
    filename0, filename1 = kwargs["filenames"]
        
    colList_float, colList_cnt, colList_days, colList_unicode = kwargs["column_lists"]   
    
    keys = kwargs["keys"]
    # Optional Args
    oversampling_ratio = kwargs["oversampling_ratio"] if "oversampling_ratio" in kwargs.keys() else 0.5
    comprehensive_search = kwargs["comprehensive_search"] if "comprehensive_search" in kwargs.keys() else False
    ###Data Loading 
    os.chdir(wkdir)
    try :
        data_ori0 = pd.read_csv(data_path + "/" + filename0 #/rex_up_features_sample0.csv"
        , low_memory=False, encoding=u'utf-8') \
            .drop_duplicates(subset='relationshipno',keep='first')
        
        data_ori1 = pd.read_csv(data_path + "/" + filename1
        , low_memory=False, encoding=u'utf-8').drop_duplicates(subset='relationshipno',keep='first')
    
        #axis = 0 means merge by column(same column join) 
        #axis = 1 means merge by row(same index join)
        data_tmp = pd.concat([data_ori0, data_ori1], axis=0)
        data_tmp.index = data_tmp[keys]
        #print(data_ori0.shape, data_ori1.shape, data_tmp.shape)
        #print(data_tmp)
        assert data_ori0.shape[0]+data_ori1.shape[0] == data_tmp.shape[0] , "0/1 Merging failed"
        assert data_ori0.shape[1] == data_ori1.shape[1] == data_tmp.shape[1] , "Column number not match"
        
        logger.info("shapes of data_ori0, data_ori1, data_tmp:" + str(data_ori0.shape) + str(data_ori1.shape) + str(data_tmp.shape))
    
        #For numeric features including float, cnt and days, we fill NaN and normalize
        #No need to discretize in this model.
        #n_disc = 5
        clients_discretized = data_tmp.loc[:, :].copy()
        #nsamples = clients_discretized.shape[0]
    
        features_num = clients_discretized[["relationshipno"]+colList_float + colList_days + colList_cnt].drop_duplicates( keep='first')
        
        features_num = features_num[colList_float + colList_days + colList_cnt] \
                         .applymap(discretize.clean_cell) \
                         .applymap(lambda x : np.float64(x)) 
        # save (mean, std) into tables so that can be retrieved at predict phase
        features_num.apply(lambda x : pd.Series([np.mean(x), np.std(x)]),axis=0).to_csv(out_path + "/" + "features_num_meanstd.csv")
        logger.info("numeric features normalization args have been writen to files:" + "features_num_meanstd.csv")
    
        features_num = features_num.apply(normalize.meanstd, axis=0).fillna(0)
                         
        logger.info("numeric features processed, shaped as : " + str(features_num.shape))
    
        features_cat=clients_discretized[["relationshipno"]+colList_unicode].drop_duplicates(keep='first')
        
        features_cat=features_cat[colList_unicode].fillna("0") \
                               .apply(lambda x: x.astype('category',ordered=True) \
                               .factorize()[0],axis=0)
        
        features_cat.apply(lambda x : pd.Series([np.mean(x), np.std(x)]),axis=0).to_csv(out_path+ "/" + "features_cat_meanstd.csv")
        
        logger.info("categorical features normalization args have been writen to files:" + "features_cat_meanstd.csv")
        
        features_cat = features_cat.apply(normalize.meanstd, axis=0) .fillna(0)
        
        logger.info("categorical features processed, shaped as : " + str(features_cat.shape))
    
        
        # Deal with label
        label_data = clients_discretized[["relationshipno"]+[labels]].drop_duplicates(keep='first')
        label_data = label_data[labels]
        
        assert sum([features_num.isnull().sum().sum(), 
                    features_cat.isnull().sum().sum() == 0])   , "There is NaN in features"
        logger.info("labels processed, shaped as : " + str(label_data.shape))
    
       
        data_all = pd.concat([features_num.loc[features_num.index.sort_values(),],
                              features_cat.loc[features_cat.index.sort_values(),], 
                              label_data.loc[label_data.index.sort_values(),]], axis=1)
        #data_all[labels] = data_all[labels].apply(lambda x: 0 if x != x else 1) 
        logger.info("merging features processed, shaped as : " + str(data_all.shape))
        assert data_all.isnull().sum().sum() == 0, "There is NaN in data_all"
    
    
        
        ###Dataset Separation###
        #oversample should be a fraction indicating how much samples oversample
        def BuildDataSet(data, X, Y, frac, keys = None, oversampling = [0, 0.5]) :
            if keys != None :
                data.index = data[keys]
                
            X_train = pd.DataFrame()
            Y_train = pd.Series()
            unique_Y = np.unique(data[Y])
            for i in range(len(unique_Y)) : 
                val_Y = unique_Y[i]
                print(val_Y)
                
                # sampling
                _X_train = data.loc[data[Y] == val_Y,X].sample(frac = frac, 
                                                 replace = False, random_state=0)
                _Y_train = data.loc[data.index.isin(_X_train.index), Y]
                
                # append Y sampling
                X_train = X_train.append(_X_train)
                Y_train = Y_train.append(_Y_train)
                
                # oversampling 
                _X_train = _X_train.sample(frac = oversampling[i], 
                                           replace=False, random_state=0)
                _Y_train = data.loc[data.index.isin(_X_train.index), Y]
                
                # append oversampling 
                X_train = X_train.append(_X_train)
                Y_train = Y_train.append(_Y_train)
         
            #X_train = X_train.loc[X_train.index.sort_values(),:]    
            #print(X_train.shape)    
                #X_train_1 = data.loc[data[Y] == 1,X].sample(frac = frac, replace = False,random_state=0)
                #print(X_train_1.shape)
                
            #Y_train = Y_train[Y_train.index.isin(X_train.index), Y]
            #Y_train.index = X_train.index
            #print(Y_train.shape)
            #Y_train_1 = data.loc[data.index.isin(X_train_1.index), Y]
            #print(Y_train_1.shape)
                
            X_test = data.loc[~data.index.isin(X_train.index),X] 
            #X_test = X_test.loc[X_test.index.sort_values(),:]    
            
            Y_test = data.loc[data.index.isin(X_test.index), Y]
            Y_test.index = X_test.index
            return X_train, Y_train, X_test, Y_test 
    
    
        
        
        ###
        #Divide data into training and testing set
        #70% training vs 30% testing
        ###
        #features =  colList_unicode + colList_int + colList_float + colList_dup
        features = data_all.columns.values.tolist()
        features.remove( labels )
        #features.remove( "osmf_hld:9.0"  )
        #labels = "osmf_hld:9.0"
        X_train, Y_train, X_test, Y_test = BuildDataSet(data_all, features, labels, 0.9,
                                                        oversampling=[0,oversampling_ratio])
        #print(X_train.head())
        
        logger.info("building dataset processed, shapes of X_train, Y_train, X_test, Y_test: " + 
                    str(X_train.shape) + str(Y_train.shape) +
                    str(X_test.shape) + str(Y_test.shape))
        
        X_test.to_csv(data_path+"_X_test.csv")
        Y_test.to_csv(data_path+"_Y_test.csv")
    
        ######################################################################
        # Train and evaluate
        ######################################################################
        def eval_pr(models, X_test, Y_test) :
            colors = ['blue','yellow','green','red','purple','pink','grey']
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([-0.01, 1.01])
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
            plt.xlim([-0.01, 1.01])
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
    
        
    
        #Fitting
        logger.info("START Training Models")
        gbdt, gbdt_par = ml.GBDT(X_train, Y_train, score='f1_micro',comprehensive=comprehensive_search)
        logger.info("GBDT gridsearch done")
        
        joblib.dump(gbdt,dump_path+"/gbdt.m")
        logger.info("Dump gbdt processed at path:" + dump_path)
            
        
        rf, rf_par = ml.RF(X_train, Y_train, score='f1_micro', comprehensive=comprehensive_search)
        logger.info("RF gridsearch done")
         
        joblib.dump(rf,dump_path+"/rf.m")
        logger.info("Dump rf processed at path:" + dump_path)
            
        svm, svm_par = ml.SVM(X_train, Y_train, score='f1_micro', comprehensive=comprehensive_search)
        logger.info("svm gridsearch done")
        
        joblib.dump(svm,dump_path+"/svm.m")
        logger.info("Dump svm processed at path:" + dump_path)
       
        lr, lr_par = ml.LR(X_train, Y_train, score='f1_micro', comprehensive=comprehensive_search)
        logger.info("lr fitting done")
        
        joblib.dump(lr,dump_path+"/lr.m")
        logger.info("Dump lr processed at path:" + dump_path)
       
        models = [("gbdt",gbdt),
                  ("rf", rf),
                  ("svm", svm),
                  ("lr", lr)]
        #Evaluate
        logger.info("START Evaluating Models")
        roc_plt, metrics_roc = eval_roc(models, X_train, Y_train)
        roc_plt.savefig(out_path + "/" + "roc_train.png")
        roc_plt.close()
        #roc_plt.show()
        
        roc_plt, metrics_roc = eval_roc(models, X_test, Y_test)
        roc_plt.savefig(out_path + "/" + "roc_test.png")
        roc_plt.close()
        #roc_plt.show()
        
        pr_plt, metrics_pr = eval_pr(models, X_train, Y_train)
        pr_plt.savefig(out_path + "/" + "pr_train.png")
        pr_plt.close()
        #pr_plt.show()
        
        pr_plt, metrics_pr = eval_pr(models, X_test, Y_test)
        pr_plt.savefig(out_path + "/" + "pr_test.png")
        #pr_plt.show()
        pr_plt.close()
    
        #Output Evaluation Reference
        for i in range(len(metrics_pr)) : 
            m = models[i]
            met = metrics_pr[i][2]
            #joblib.dump(m[1],dump_path+"/"+m[0]+".m")
            #logger.info("Dumping " + m[0] + " processed at path:" + dump_path)
            
            precision, recall, thresholds, x = met
            rslt = np.concatenate((precision.reshape(precision.shape[0],1)[1:,:]
                             ,recall.reshape(recall.shape[0],1)[1:,:]
                             ,thresholds.reshape(thresholds.shape[0],1)
                            ), axis=1)
            rslt = pd.DataFrame(rslt, columns=["precision","recall","threshold"])
            plt.figure()
            plt.plot(rslt["threshold"], rslt["precision"], color='blue')
            plt.plot(rslt["threshold"], rslt["recall"], color='green')
     
            plt.savefig(out_path + "/" + m[0] + "_threshold.png")
            plt.close()
            
            pr_matrix = pd.DataFrame()
            for th in [0.1, 0.2, 0.3, 0.4,
                       0.5, 0.6, 0.7, 0.8, 
                       0.9, 0.95, 0.99, 0.995,
                       0.999, 0.9999] :
                pr_matrix = pr_matrix.append( rslt.loc[rslt["threshold"] >= th, :].head(1) )
                
            pr_matrix.to_excel(out_path+"/" + m[0] + "_score_matrix.xlsx", index=None)
    
            #Feature Importance
            feature_importance = pd.DataFrame([X_test.columns,gbdt.feature_importances_]).T
            fi = feature_importance.rename(columns={0:"feature", 1:"importance"}).sort_values(by="importance",ascending=False)
            
            fi.to_excel(out_path + '/' + m[0] + '_feature_importance.xlsx', index=False, merge_cells=False)
            
            logger.info("Finish") 
    except Exception as err:
        
        logger.error(str(err))
    finally:
        logger.removeHandler(handler)
        
    return models, metrics_pr


if __name__ == "__main__" :
    os.chdir(wkdir)
    models = train(path_list=[wkdir, dump_path, data_path, out_path],
          filenames=["", ""],
          column_lists=[colList_float, colList_cnt, colList_days, colList_unicode],
          keys = '',
          comprehensive_search=True)
    
