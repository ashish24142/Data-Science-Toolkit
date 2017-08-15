# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:30:19 2017

@author: Ashish Kr Singh
"""

# Importing the libraries
import numpy as np
import pandas as pd
import scipy as sc
import math
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder

def gini(data, class_values):
    unique_values = np.unique(data)
    lists = pd.DataFrame(
    {'data': data,
     'class': class_values
    })
    
    total_size = len(lists)
    valuess = []
    for values in unique_values:
        s = lists[lists['data'] == values]
        a = np.array(s['class'])
        sums = (a==1).sum()
        prob = float(float(sums)/float(total_size))
        valuess = np.append(valuess,prob)
     
    total_prob_sq = 0
    for value in valuess:
        total_prob_sq = float(total_prob_sq + float(value * value))
    
    return float(1-total_prob_sq)
        
        

def accuracy(data, class_values):
    unique_values = np.unique(data)
    lists = pd.DataFrame(
    {'data': data,
     'class': class_values
    })
    
    total_size = len(lists)
    valuess = []
    for values in unique_values:
        s = lists[lists['data'] == values]
        a = np.array(s['class'])
        sums = (a==1).sum()
        prob = float(float(sums)/float(total_size))
        valuess = np.append(valuess,prob)
        
    return np.max(valuess) 

def entropy(data, class_values):
    
    unique_values = np.unique(data)
    lists = pd.DataFrame(
        {'data': data,
         'class': class_values
        })
     
    k = float(len(unique_values))
    total_size = len(lists)
    valuess = []
    for values in unique_values:
        s = lists[lists['data'] == 1]
        a = np.array(s['class'])
        sums = (a==1).sum()
        prob = float(float(sums)/float(total_size))
        valuess = np.append(valuess,prob)
         
    total_entropy = 0.0
    for value in valuess:
        if (value == 0):
            total_entropy = total_entropy + 0
        else:
            value
            calculated_value = float(value * (math.log(value, k)))
            total_entropy = float(total_entropy + calculated_value)    
    
    return float(1-((-1.0)*total_entropy))
# Importing the dataset

# Read Dataset
dataset = pd.read_csv('\\train.csv')

# Encode Categorical Values of Dataset
labelencoder_X = LabelEncoder()
newdf = dataset[dataset.columns[1:24]]
en_df = newdf.apply(LabelEncoder().fit_transform)
x = en_df.iloc[:,].values

# Encoding the Dependent Variable
df = dataset[dataset.columns[2:24]]

total_features = list(df)            

# Construct Table of Feature Name, Accuracy, Gini Index, `1-Entropy
columns = ['Feature_Name','Accuracy', 'Gini_Index', '1_Entropy']
df_ = pd.DataFrame(columns=columns)

i = 0
for feature in total_features:
    feature_name = total_features[i]
    gini_index = gini(x[:,i+1], x[:,0])
    entropy_f = entropy(x[:,i+1], x[:,0])
    accuracy_f = accuracy(x[:,i+1], x[:,0])
    new_frame = pd.DataFrame([[feature_name,accuracy_f,gini_index,entropy_f]], columns=columns)
    df_ = df_.append(new_frame)
    i = i+1

feature = df_.iloc[:,0].values
accuracy = df_.iloc[:,1].values 
entropy = df_.iloc[:,3].values 
                   
                   
plt.scatter(accuracy, entropy)
plt.show()