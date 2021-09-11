# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:15:42 2020

@author: Steven Ho

Input data:    
Position of horse , Loading of horse, Odds (Overnight), Odds (15 min before the race), Class of horses, Num of horses in race

Model v1.0 (2020/12/02): 
    -inital release 
    
Model v1.1 (2021/02/05):
    -changed activation function of input and hidden layer = "relu"
    -changed optimizer to "adam"
    
Model v1.2 (2021/09/11):
    -changed the size of hidden layer = 10

"""

import pandas as pd
import tensorflow as tf
tf.__version__

#defines

NUM_OF_EPOCHS = 50000 #20000
PATH_TRAINING_DATA = 'horse_data_train_test.csv'

# Part 1 - Data Preprocessing

dataset = pd.read_csv(PATH_TRAINING_DATA)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle=False)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #use "fit_transform" for training data
X_test = sc.transform(X_test)       #use "transform" for testing data

y_train = tf.keras.utils.to_categorical((y_train-1), num_classes=None)
y_test = tf.keras.utils.to_categorical((y_test-1), num_classes=None)

print(y_train)

# Part 2 - Building the ANN

# Initializing the ANN
model = tf.keras.models.Sequential()

# Adding the input layer and the first layer
# We have six features now, so the input layer has a size of 6
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the input layer and the hidden layer
# We set the size of the hidden layer to be the mean of input and output later, i.e. 10
model.add(tf.keras.layers.Dense(units=10, activation='relu')) #model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
# We have 14 outputs, so the output later has a size of 14
model.add(tf.keras.layers.Dense(units=14, activation='softmax'))

# Part 3 - Training the ANN

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Training the ANN on the Training set
print("Training model...")
model.fit(X_train, y_train, batch_size = 14, epochs = NUM_OF_EPOCHS)

# save the model for later use
print("Saving model...")
model.save('saved_model/my_model')

print("Training Finished!")
