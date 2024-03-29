# -*- coding: utf-8 -*-
"""hmmconfig.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z47g4lwE3N_YFWhHYv8Hc_XT_H_s03bD
"""

def startprobability():
 import numpy as np  
 startprob = np.array([0.5, 0.3, 0.2])
 return startprob

def transmatrix():
 import numpy as np
 transmat= np.array([[0.5, 0.3, 0.2],
                      [0.3, 0.5, 0.2],
                      [0.2, 0.3, 0.5]])
 return transmat

def emissionprobability():
 import numpy as np
 emissionprob = np.array([[0.5, 0.3, 0.2],
                      [0.3, 0.5, 0.2],
                      [0.2, 0.3, 0.5]])
 return emissionprob