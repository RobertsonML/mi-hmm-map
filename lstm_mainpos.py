# -*- coding: utf-8 -*-
"""lstm_ARIMA_SC-MURDERS2013_2015.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TgzJB5mm-EB_Was_8TLjWIIKJpcwSAOq
"""

import tensorflow as tf
import os
import pandas as pd
import numpy as np

!cp /content/drive/MyDrive/Colab\ Notebooks/ePredictor/preprocess.py /content

import preprocess
df=preprocess.getfile()

df.index = pd.to_datetime(df['MonthYear'], format='%d/%m/%Y')
df[:24]

temp = df['SeriousCrimes']
temp.plot() #grab Homicide Offenses

def df_to_X_y(df, window_size=6):
 df_as_np = df.to_numpy()
 X =[]
 y =[]
 for i in range(len(df_as_np)-window_size):
  row = [[a] for a in df_as_np[i:i+6]]
  X.append(row)
  label = df_as_np[i+6]
  y.append(label)
 return np.array(X), np.array(y)

window_size =6
X, y = df_to_X_y(temp, window_size)
X.shape, y.shape

X_train, y_train = X[:24], y[:24]
X_val, y_val = X[12:24], y[12:24]
X_test, y_test = X[24:36], y[30:]

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam

model1 =Sequential()
model1.add(InputLayer((6, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear')) #result linear

model1.summary()

cp = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), metrics = [RootMeanSquaredError()])

model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300,  verbose = 0, callbacks = [cp])

from keras.models import load_model
model1 = load_model('model1/') #loads one with lowest value into memory

train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
train_results

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'])
plt.plot(train_results['Actuals'])
plt.xlabel("Months")
plt.ylabel("State")
plt.legend(['Prediction', 'Actual'])
plt.title("Arima (Trinidad W.I.) Police Bureau")

test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
test_results

plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])

forecast_errors = [y_test[i]-test_predictions[i] for i in range(len(y_test))]
print('Forecast Errors: %s' % forecast_errors)

mean_forecast_error = sum(forecast_errors)/12
print('Mean_Forecast_Error: %f' % mean_forecast_error)

predval = list(range(len(test_predictions)))
for i in range(0,len(test_predictions)):
  if (test_predictions[i] < 0.5):
   predval[i] = 0
  elif (test_predictions[i] < 1.5):
   predval[i] = 1
  else:
   test_predictions[i] = 2
test_predictions= predval

test_predictions

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, test_predictions)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, test_predictions, pos_label='positive', average='weighted')
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, test_predictions, pos_label='positive', average='weighted')
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, test_predictions, pos_label='positive', average='weighted')
print('F1 score: %f' % f1)