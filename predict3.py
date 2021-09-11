# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:03:56 2020

@author: Steven Ho
"""

# Restore the weights
import numpy as np
import pandas as pd

PATH_TRAINING_DATA = 'horse_data_train_test.csv'

CURRENT_RACE_DATA = 'new_data/horse_data_20210908_race2.csv'

import tensorflow as tf

#training data
dataset = pd.read_csv(PATH_TRAINING_DATA)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle=False)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#live data
dataset = pd.read_csv(CURRENT_RACE_DATA)
X_live = dataset.iloc[:, :-1].values

#use "transform" for live data
X_live = sc.transform(X_live) #X_live = sc.fit_transform(X_live)

#loading pretrained model
model = tf.keras.models.load_model('saved_model/my_model')

# Predicting the Test set results
y_pred = model.predict(X_live)  #y_pred = model.predict( sc.fit_transform(X_live)) 
#print("y_pred=", y_pred)

# Predicting the finishing position
y_pred_finishing = np.argmax(y_pred, axis = 1) 
print("Expected finishing positions=", y_pred_finishing+1)

