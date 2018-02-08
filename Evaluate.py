# Define a model and then plot the PR curve and ROC curve
def Evalueate(model) :
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
