#external data science packages
import numpy as np
import pandas as pd 
from sklearn import tree 
from sklearn.externals import joblib

#external plotting packages
import matplotlib.pyplot as plt

#system packages
import os
import logging

from sys import argv

from datetime import datetime

#User defined packages
from preprocessing import hash_col

from preprocessing import onehot

from preprocessing import normalize

from preprocessing import discretize

from input_args import *
    

#------------------------------------------------------------------------------
#predict : overall workflow of prediction phase, after the models are already trained.
def predict(*args , **kwargs) :

    
    # Mandatory Args
    wkdir, dump_path, data_path, out_path = kwargs["path_list"]
   
    file_p, file_n, file_c = kwargs["filenames"]
        
    colList_float, colList_cnt, colList_days, colList_unicode = kwargs["column_lists"]   

    model_name = kwargs["model_name"]
    
    keys = kwargs["keys"]
    # Optional Args

    # Setup paths and start logging
    os.chdir(wkdir)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info("START Predicting")
    
    
    try:
    
        ###Data Loading 
        data_pred = pd.read_csv(data_path + "/" + file_p
        , low_memory=False, encoding=u'utf-8') .drop_duplicates(subset=keys,keep='first')
        data_pred.index = data_pred[keys]
        
        meanstd_n = pd.read_csv(data_path + "/" + file_n)    
        meanstd_c = pd.read_csv(data_path + "/" + file_c)
        
        #For numeric features including float, cnt and days, we fill NaN and normalize
        clients_discretized = data_pred.copy()
     
        features_num = clients_discretized[[keys]+colList_float + colList_days + colList_cnt].drop_duplicates( keep='first')
        
        
        features_num = features_num[colList_float + colList_days + colList_cnt] \
                     .applymap(discretize.clean_cell) \
                     .applymap(lambda x : np.float64(x)) \
                     .apply(lambda x : (x - meanstd_n[x.name][0]) / (meanstd_n[x.name][1]+1e-5),axis=0) \
                     .fillna(0)   
                    
        logger.info("numeric features processed, shaped as : " + str(features_num.shape))
    
        features_cat=clients_discretized[[keys]+colList_unicode].drop_duplicates(keep='first')
        
        features_cat=features_cat[colList_unicode].fillna("0") \
                               .apply(lambda x: x.astype('category',ordered=True) \
                               .factorize()[0],axis=0) \
                               .apply(lambda x : (np.float64(x) - meanstd_c[x.name][0]) /(meanstd_c[x.name][1]+1e-5),axis=0) \
                               .fillna(0)  
           
        logger.info("categorical features processed, shaped as : " + str(features_cat.shape))
    
        
        assert sum([features_num.isnull().sum().sum(), 
                    features_cat.isnull().sum().sum() == 0])   , "There is NaN in features"
        
        data_all = pd.concat([features_num.loc[features_num.index.sort_values(),],
                              features_cat.loc[features_cat.index.sort_values(),]], 
                                axis=1)
        
        logger.info("merging features processed, shaped as : " + str(data_all.shape))
        assert data_all.isnull().sum().sum() == 0, "There is NaN in data_all"
       
        logger.info("data_all built with shape:"+str(data_all.shape))
        
        # Double confirm no NaN exists after merging
        assert data_all.isnull().sum().sum() == 0 , "There is NaN in data_all"
        #print(data_all.isnull().sum().sum())
        # Get feature names
        features = data_all.columns.values.tolist()
        
        # Load pre-trained model, GBDT by default
        clf = joblib.load(dump_path+"/" + model_name + ".m")
        logger.info("model loaded from :" + dump_path+"/" + model_name + ".m")

        #rf   = joblib.load("dumped_models/rf.m")
        data_prd = data_all[features]
        #data_prd.index = data_all.index
        # 
        pred_prd = clf.predict_proba(data_prd)
    
        # Prediction output will be an list of pairs, the position 1 indicates the Class-1
        # If the prediction distribution is abnormal, check this 0/1 positions
        pred_df = pd.DataFrame(pred_prd[:,1], index = data_prd.index, columns=['pred'])
        output = pred_df
        logger.info("Scoring done")
        
        # Reorganize as an output format
        # output predicted scores and some basic information. '
        # Got information from original dataset without transforming
        # That's why we need a copy of data_pred. 
        data_tmp2 = data_pred.drop_duplicates(keep='first')
        output_col = []
		
        output    = data_tmp2.loc[pred_df.index, output_col]
        output    = pd.concat([pred_df,  output], axis=1)
        #output.to_csv(out_path + "/pred_all.csv")
        output.to_excel(out_path + "/pred_all.xlsx")
                
        logger.info("Output with common features to: " + out_path + "/pred_all.xlsx")
        #logger.info("Prediction Succeeded.")
        
        logger.info("Start plotting histogram of scores")
        #Prediction Histogram
        thres = 0.5
        hist, bins = np.histogram(np.array(output.loc[output["pred"] > thres, "pred"]), bins=20, density=False)
        width = 0.5 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        
        def fivenum(x) :
            return [ round(np.min(x),2)
                    , round(np.percentile(x, 25),2)
                    , round(np.median(x),2)
                    , round(np.percentile(x, 75),2)
                    , round(np.max(x),2)]
            
        plt.title("fivenum are {0[0]:.2}, {0[1]:.2}, {0[2]:.2}, {0[3]:.2}, {0[4]:.2}".format(fivenum(output.loc[output["pred"]>thres, "pred"])))
        plt.savefig(out_path + "/score_histogram.png")
        plt.close()
        logger.info("Plotting done")
        logger.info("Prediction process finished")
        
                
        #cut off clients
        def clientsAtCutoff(output, cutoff, colname = "pred") :
            clients = []
            for cf in cutoff : 
                clients.append( output.loc[output[colname] > cf , [colname]].shape[0] )
            return clients
        cutoffs = [0, 0.1, 0.2, 0.3, 0.4, 
                   0.5, 0.6, 0.7, 0.8,
                   0.9, 0.95, 0.97, 0.99,
                   0.995, 0.999, 0.9999]
        clients_at = clientsAtCutoff(output, cutoffs)
        print("Clients at cutoffs are:" + str(clients_at))
        pd.Series(clients_at, index=cutoffs).to_excel(out_path+"/score_distr.xlsx")
    except Exception as err:
        logger.error(str(err))
        
        #print(err)
    finally:
        logger.removeHandler(handler)

if __name__ == '__main__' :
    wkdir = u''

    #set parameters 
    dump_path = wkdir + "/model"
    data_path = wkdir + "/data"
    out_path  = wkdir + "/output"
    
    colList_float = []
    
    colList_cnt = []
    
    colList_days = []
    
    colList_unicode = []
    model_name=''
    keys = ''
    
    os.chdir(wkdir)
    tic = datetime.now()
    predict(path_list=[wkdir, dump_path, data_path, out_path],
          filenames=["", "", ""],
          column_lists=[colList_float, colList_cnt, colList_days, colList_unicode],
          model_name=model_name, keys=keys)
    toc = datetime.now()
    print("predict time:" + str(toc-tic))
