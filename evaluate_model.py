# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:00:40 2020

@author: Steven Ho
"""

#This needs to combine with the old ANN implementation

# Restore the weights
import numpy as np
import pandas as pd

dataset = pd.read_csv('horse_data_train_test.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


import tensorflow as tf
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


new_model = tf.keras.models.load_model('saved_model/my_model')

# Predicting the Test set results
y_pred = new_model.predict( sc.fit_transform(X_test) ) 
y_pred = np.argmax(y_pred, axis = 1)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, (y_pred+1))
print("cm=", cm)
print("accuracy_score=", accuracy_score(y_test, (y_pred+1)))