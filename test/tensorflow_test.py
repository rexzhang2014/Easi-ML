# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
#%%
iris = load_iris()
col_X = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
col_Y = ['Species']

iris_X = pd.DataFrame(iris.data, columns=col_X)
iris_Y = pd.DataFrame(iris.target, columns=col_Y)

iris_data = pd.concat([iris_X, iris_Y], axis=1)

#%%
###Dataset Separation###
def BuildDataSet(data, X, Y, frac, keys = None) :
    if keys != None :
        data.index = data[keys]

    X_train = pd.DataFrame()
    for val_Y in np.unique(data[Y]) :

        X_train = X_train.append(data.loc[data[Y] == val_Y,X]
                                 .sample(frac = frac,
                                         replace = False,
                                         random_state=0))


    print(X_train.shape)
        #X_train_1 = data.loc[data[Y] == 1,X].sample(frac = frac, replace = False,random_state=0)
        #print(X_train_1.shape)

    Y_train = data.loc[data.index.isin(X_train.index), Y]
    print(Y_train.shape)
    #Y_train_1 = data.loc[data.index.isin(X_train_1.index), Y]
    #print(Y_train_1.shape)

    X_test = data.loc[~data.index.isin(X_train.index),X]
    Y_test = data.loc[data.index.isin(X_test.index), Y]

    return X_train, Y_train, X_test, Y_test


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels, batch_size) :
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
    #return dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


#%%
BATCH_SIZE = 10
TRAIN_STEPS = 100

my_feature_columns = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]


classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)


#%%
X_train, Y_train, X_test, Y_test = BuildDataSet(iris_data, col_X, col_Y[0],
                                                frac=0.7)


#%%

classifier.train(
        input_fn=lambda:train_input_fn(X_train, Y_train, BATCH_SIZE),
        steps=TRAIN_STEPS)
#%%
# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(X_test, Y_test, BATCH_SIZE))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
#%%
# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x, batch_size=BATCH_SIZE))
#%%
for pred_dict, expec in zip(predictions, expected):
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(iris_data.SPECIES[class_id],
                          100 * probability, expec))
#%%
from pyhive import hive
conn = hive.Connection(host="cnlpaphdp02", port=10000, username="1542483", password='Tongji1907.')

df = pd.read_sql("SELECT * FROM cnedmp.ebbs_cust where dt=20180330 limit 10", conn)
