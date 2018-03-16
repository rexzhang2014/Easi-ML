# Train and evaluate
from sklearn import metrics

def Evalueate(model, X_test, Y_test) :
    pred = gbdt.predict_proba(X_test)
    #print(pred)
    precision, recall, thresholds = metrics.precision_recall_curve(Y_test, pred[:,1])
    
    average_precision = metrics.average_precision_score(Y_test, pred[:,1])
    
    
     
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
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
#%% Fitting
    
gbdt = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01,
   max_depth=5, random_state=0, max_features='auto').fit(X_train, Y_train)

#%% Evaluate    
#Evalueate(gbdt ,X_train ,Y_train)

precision, recall, thresholds = Evalueate(gbdt ,X_test ,Y_test)

# Find Cutoff by precision(PR)

#rslt1 = None
#xxx = None
#xxx1 = None
#cutoff = rslt.loc[xxx1.iloc[:,0] > 0.95,:]
cutoff = rslt[["threshold","precision","recall"]]
#print(thresholds[min(rslt1.index)],precision[min(rslt1.index)],recall[min(rslt1.index)])
print(cutoff.loc[cutoff["threshold"] >= 0.8,:].head())

##########################################################################
##for reference
rf = RandomForestClassifier(n_estimators=500, max_depth=20)
rf = rf.fit(X_train, Y_train)

cart = tree.DecisionTreeClassifier(min_samples_split=3)
cart = cart.fit(X_train, Y_train)

pred = rf.predict_proba(X_test)
