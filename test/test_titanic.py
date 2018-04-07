# -*- coding: utf-8 -*-
#%%
import pipline.framework as fw
import pandas as pd
import os

os.chdir("F:\Project\GitClone\EasiML")
#%%
data_ori = pd.read_csv("data/titanic_train.csv", low_memory=False, encoding=u'utf-8')
data_pred = pd.read_csv("data/titanic_test.csv")

#%%
cat_features = ['Pclass'
                ,'Name'
                ,'Sex'
                ,'Ticket'
                ,'Cabin'
                ,'Embarked']

con_features = ['Age'
,'SibSp'
,'Parch'
,'Fare']

label = ["Survived"]

params = dict()
params["con_features"] = con_features
params["cat_features"] = cat_features
params["label"]        = label
params["n_disc"]       = 3
params["frac"]         = 0.8
params["sparse"]       = False
params["dropNaN"]      = True
params["way_disc"]     = "equal_width"
params["score"]        = 'roc_auc' #'accuracy'
#%%


model, onehot_names, datasets = fw.EasiML_Modeling(data_ori.loc[:,:], params)

params["onehot_names"] = onehot_names

#%%
pred  = fw.EasiML_predict(model, data_pred.iloc[:,:], params)
#print(pred)

X1 = pd.DataFrame((pred[:,0]-pred[:,1]).astype(int), columns=['Survived'])
X1.index = data_pred['PassengerId']
#rlt = pd.concat([data_pred['PassengerId'], X1], axis=1)
X1.to_csv('data/submission.csv', index_label='PassengerId')
#model = fw.EasiML_Modeling(data_ori.iloc[:1000,:], params)
#pred  = fw.EasiML_predict(model, data_ori.iloc[:1000,:])
#%%
#%%
import tensorflow as tf
BATCH_SIZE = 10
TRAIN_STEPS = 100

for k in datasets.X
my_feature_columns = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]


#import tensorflow.estimator.inputs.numpy_input_fn
classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[15, 10, 10],
        n_classes=3)


classifier.train(
        #input_fn=lambda:tf.estimator.inputs.pandas_input_fn(X_train, Y_train, shuffle=True, batch_size=BATCH_SIZE),
        input_fn=lambda:train_input_fn(X_train, Y_train, batch_size=BATCH_SIZE),
        steps=TRAIN_STEPS)

eval_result = classifier.evaluate(
    #input_fn=lambda:tf.estimator.inputs.pandas_input_fn(X_train, Y_train, shuffle=True, batch_size=BATCH_SIZE)
    input_fn=lambda:eval_input_fn(X_test, Y_test, batch_size=BATCH_SIZE)
    )
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
