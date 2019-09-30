from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dataset=pd.read_csv('PDE_R2_0.csv')
y=dataset['bug_cnt']
dataset.drop(['bug_cnt'], axis=1, inplace=True)
x_columns = dataset.columns
y_columns = ['bug_cnt']

dataset = np.array(dataset)
y= np.array(y)

kf = KFold(n_splits=9,shuffle=False,random_state=None)

i = 1
for train_index, test_index in kf.split(dataset):
    print('TRAIN', train_index, 'TEST', test_index)
    train_x, train_y = dataset[train_index], y[train_index]
    test_x, test_y = dataset[test_index], y[test_index]

    pd.DataFrame(train_x, columns=x_columns).to_csv("PDE_R2_0-data/fold{}_train_x.csv".format(i), index=False)
    pd.DataFrame(train_y, columns=y_columns).to_csv("PDE_R2_0-data/fold{}_train_y.csv".format(i), index=False)
    pd.DataFrame(test_x, columns=x_columns).to_csv("PDE_R2_0-data/fold{}_test_x.csv".format(i), index=False)
    pd.DataFrame(test_y, columns=y_columns).to_csv("PDE_R2_0-data/fold{}_test_y.csv".format(i), index=False)
    i+=1
