# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:03:04 2016

@author: jiamingliu
"""
import pickle
import pandas as pd
import sys

path = '/Users/YaqingXie/Desktop/BDA_Project'

image_name = sys.argv

#input filename here    
filename = path+'/output/feature/' + image_name[1] + '_feature7.csv'

#read the random forest model
with open(path+'/lib/random forest.pkl', 'rb') as f:
    rfc = pickle.load(f)
    
#preprocess test data
df = pd.read_csv(filename)
data_cols = [col for col in df.columns if col not in ['id']]
x_test = df[data_cols].values

#predict test data
predict_label = rfc.predict(x_test).tolist()

#store label into dataframe
label = pd.Series(predict_label)
df['label'] = label.values

#add definition
defi = pd.read_csv(path+'/data/label_definition.csv')

#output file
final = pd.merge(left=df, right=defi, how='left')
final[['label', 'definition']].to_csv(path+'/output/prediction/' + image_name[1] + '_prediction.csv', index=False)
