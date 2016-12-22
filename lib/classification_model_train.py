# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:16:26 2016

@author: jiamingliu
"""
#%% k nearest neighbor classifier
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


df_img = pd.read_csv('driver_imgs_list.csv')
df = pd.read_csv('feature8.csv')
df_img = df_img.drop('classname',axis = 1)

df_all = pd.DataFrame.merge(df,df_img,left_on = 'id',right_on = 'img')
data_cols = [col for col in df_all.columns if col not in ['img', 'id']]
feature_cols = [col for col in df_all.columns if col not in ['img','id','class','class_01']]
label_cols = ['class']

dataset = df_all[data_cols]

train=dataset.sample(frac=0.8,random_state=200)
test=dataset.drop(train.index)

x_train, x_test, y_train, y_test = train_test_split(dataset[feature_cols].values, dataset[label_cols].values, test_size=0.2, random_state=0)
'''
x_train = train[feature_cols].values
y_train = train[label_cols].values




#x_test = test[feature_cols].values
#y_test = test[label_cols].values
test_df = pd.read_csv('feature9_80.csv')
test_img = pd.read_csv('test.csv')

test_all = pd.DataFrame.merge(test_df,test_img,left_on = 'id',right_on = 'img')
x_test = test_all[feature_cols].values
y_test = test_all[label_cols].values
'''


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train, y_train[:,0]) 
score = neigh.score(x_test,y_test[:,0])

import pickle
with open('k nearest neighbor.pkl', 'wb') as fout:
  pickle.dump(neigh, fout)


primary_label = y_test.ravel().tolist()
predict_label = neigh.predict(x_test).tolist()
diff = [primary_label[i]-predict_label[i] for i in range(len(primary_label))]
percentage = diff.count(0)/len(primary_label)
#print(y_test[:,0])
#print(neigh.predict(x_test))

#df_all.to_csv('feature9_all.csv')
#kmeans = KMeans(n_clusters=10, random_state=0).fit(x)
#sk.cluster(10,30,)

#%% neural network classifier


import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df_img = pd.read_csv('driver_imgs_list.csv')
df = pd.read_csv('feature8.csv')
df_img = df_img.drop('classname',axis = 1)

df_all = pd.DataFrame.merge(df,df_img,left_on = 'id',right_on = 'img')
data_cols = [col for col in df_all.columns if col not in ['img', 'id']]
feature_cols = [col for col in df_all.columns if col not in ['img','id','class','class_01']]
label_cols = ['class']

dataset = df_all[data_cols]

x_train, x_test, y_train, y_test = train_test_split(dataset[feature_cols].values, dataset[label_cols].values, test_size=0.2, random_state=0)
'''

train=dataset.sample(frac=0.8,random_state=200)
test=dataset.drop(train.index)

x_train = train[feature_cols].values
y_train = train[label_cols].values


x_test = test[feature_cols].values
y_test = test[label_cols].values
'''
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 2), random_state=1)
clf.fit(x_train, y_train[:,0])
score = clf.score(x_test,y_test[:,0])
import pickle
with open('neural network.pkl', 'wb') as fout:
  pickle.dump(clf, fout)



primary_label = y_test.ravel().tolist()
predict_label = neigh.predict(x_test).tolist()
diff = [primary_label[i]-predict_label[i] for i in range(len(primary_label))]
percentage = diff.count(0)/len(primary_label)


#%% random forest classifier


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df_img = pd.read_csv('driver_imgs_list.csv')
df = pd.read_csv('feature7.csv')
df_img = df_img.drop('classname',axis = 1)



df_all = pd.DataFrame.merge(df,df_img,left_on = 'id',right_on = 'img')
data_cols = [col for col in df_all.columns if col not in ['img', 'id']]
feature_cols = [col for col in df_all.columns if col not in ['img','id','class','class_01']]
label_cols = ['class']

dataset = df_all[data_cols]
#dataset = pd.read_csv('train_data.csv')

x_train, x_test, y_train, y_test = train_test_split(dataset[feature_cols].values, dataset[label_cols].values, test_size=0.2, random_state=32)


#print(dataset[feature_cols].values)
'''

train=dataset.sample(frac=0.8,random_state=200)
test=dataset.drop(train.index)


x_train = train[feature_cols].values
y_train = train[label_cols].values


#x_test = test[feature_cols].values
#y_test = test[label_cols].values

test_df = pd.read_csv('feature9_80.csv')
test_img = pd.read_csv('test.csv')

test_all = pd.DataFrame.merge(test_df,test_img,left_on = 'id',right_on = 'img')
x_test = test_all[feature_cols].values
y_test = test_all[label_cols].values
'''


rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(x_train, y_train[:,0]) 
score = rfc.score(x_test,y_test[:,0])

import pickle
with open('random forest.pkl', 'wb') as fout:
  pickle.dump(rfc, fout)


primary_label = y_test.ravel().tolist()
predict_label = rfc.predict(x_test).tolist()
diff = [primary_label[i]-predict_label[i] for i in range(len(primary_label))]
percentage = diff.count(0)/len(primary_label) 
#print(y_test[:,0])
#print(rfc.predict(x_test))

#df_all.to_csv('feature9_all.csv')
#kmeans = KMeans(n_clusters=10, random_state=0).fit(x)


