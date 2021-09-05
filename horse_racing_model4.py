# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:15:42 2020

@author: Steven Ho

Input data:
    
position, load, ON (Overnight) odds, odds, "class", num of horses in race

"""

# Artificial Neural Network

# Importing the libraries

import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('horse_data_train_test.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(X)
#print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle=False)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#from tf.keras.utils import to_categorical
y_train = tf.keras.utils.to_categorical((y_train-1), num_classes=None)
y_test = tf.keras.utils.to_categorical((y_test-1), num_classes=None)

print(y_train)

# # Part 2 - Building the ANN

# # Initializing the ANN
model = tf.keras.models.Sequential()

# # Adding the input layer and the first layer
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# # Adding the input layer and the hidden layer
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# # Adding the output layer
model.add(tf.keras.layers.Dense(units=14, activation='softmax'))

# # Part 3 - Training the ANN

# # Compiling the ANN
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Training the ANN on the Training set
model.fit(X_train, y_train, batch_size = 14, epochs = 50000)

model.save('saved_model/my_model')




