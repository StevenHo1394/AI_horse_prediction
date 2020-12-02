# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:03:56 2020

@author: longi
"""

# Restore the weights
import numpy as np
import pandas as pd

dataset = pd.read_csv('horse_data_20201202_race2.csv')
X_live = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_live = sc.fit_transform(X_live)

model = tf.keras.models.load_model('saved_model/my_model')

# Predicting the Test set results
y_pred = model.predict( sc.fit_transform(X_live)) 
#print("y_pred=", y_pred)

# Predicting the winning horse
y_pred_winning_horse = np.argmax(y_pred, axis = 1) 
print("Expected finishing postions=", y_pred_winning_horse+1)

