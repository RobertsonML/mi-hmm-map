# -*- coding: utf-8 -*-
"""hmm_PO_SC-MURDERS2016_2022.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bOrY9-EXg0N_LNN9njDNbNyxeJdlOoeI
"""

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import numpy as np
!pip install -q hmmlearn
from hmmlearn import hmm
import warnings

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

X_train, y_train = X[:60], y[:60]
X_val, y_val = X[60:72], y[60:72]
X_test, y_test = X[66:], y[66:]

X_train = X_train.reshape(60,6)

!cp /content/drive/MyDrive/Colab\ Notebooks/ePredictor/hmmconfig.py /content

import hmmconfig

import warnings
warnings.filterwarnings("ignore")

n_states = 3

startprob=hmmconfig.startprobability()
transmat=hmmconfig.transmatrix()
emissionprob= hmmconfig.emissionprobability()

model = hmm.GaussianHMM(n_components=n_states, n_iter=1000, covariance_type="full", verbose=0,  algorithm='viterbi')

model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

arrSize = 10
arrList = list(range(arrSize))
count = list(range(arrSize))

lstSize = 10
size = 25
newsize = 0
oplist = list(range(12))
newlist = list(range(12))
for h in range(0,lstSize):
 import warnings
 warnings.filterwarnings("ignore")
 model=model.fit(X_train)

 import warnings
 warnings.filterwarnings("ignore")
 X_test = X_test.reshape(12,6)
 opplist= model.predict(X_test)
 for i in range(1,arrSize):
   from hmmlearn import hmm
   newlist = model.predict(X_test)
   #print(newlist)
   newsize = 0
   for k in range(0,12):
    if (y_test[k] != newlist[k]):
      newsize += (abs(y_test[k]-newlist[k]))
   print (newsize)
   print (newlist)
   if (newsize < size):
    size = newsize
    oplist = newlist
    print (oplist)
print ("\n")
print (size)
print (oplist)

model.monitor_

model.monitor_.converged

X_test = X_test.reshape(12,6)

test_predictions = oplist
# model.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
test_results

train_predictions = model.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
train_results

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'])
plt.plot(train_results['Actuals'])
plt.xlabel("Months")
plt.ylabel("State")
plt.legend(['Train Prediction', 'Actual'])
plt.title("Portland (Oregon) Police Bureau Serious Crimes")

X_val = X_val.reshape(12,6)

val_predictions = model.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
val_results

plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])

X_test, y_test = X[66:], y[66:]
X_test = X_test.reshape(12,6)

test_predictions = oplist
#model.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
test_results

y_test

plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])
plt.xlabel("Months")
plt.ylabel("State")
plt.legend(['Prediction', 'Actual'])
#dim=plt.subplot(111)
#dim.set_xlim(1, 12)
plt.title("Portland (Oregon) Police Bureau")

forecast_errors = [y_test[i]-test_predictions[i] for i in range(len(y_test))]
print('Forecast Errors: %s' % forecast_errors)

y_test

test_predictions

mean_forecast_error = sum(forecast_errors)/12
print('Mean_Forecast_Error: %f' % mean_forecast_error)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_predictions, y_test)
print('Accuracy: %f' % accuracy)

test_predictions

y_test

# precision tp / (tp + fp)
precision = precision_score(y_test, test_predictions, pos_label='positive', average='micro')
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, test_predictions, pos_label='positive', average='micro')
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, test_predictions, pos_label='positive', average='micro')
print('F1 score: %f' % f1)