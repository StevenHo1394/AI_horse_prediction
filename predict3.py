# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:03:56 2020

@author: Steven Ho
"""

# Restore the weights
import numpy as np
import pandas as pd

CURRENT_RACE_DATA = 'new_data/horse_data_20210203_race7.csv'

dataset = pd.read_csv(CURRENT_RACE_DATA)
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

# Predicting the finishing position
y_pred_finishing = np.argmax(y_pred, axis = 1) 
print("Expected finishing positions=", y_pred_finishing+1)

