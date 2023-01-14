from PRAN import PRAN

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_csv('~/Documents/PRAN/src/data/hepatitis.csv')

mmscaler = MinMaxScaler()
scaled_features = mmscaler.fit_transform(df.drop(["Class"], axis = 1))
X = scaled_features
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=1, stratify = y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify = y_train)


pran = PRAN(X_train, y_train, X_val, y_val, 50)
pran.fit(epochs = 100, batch_size = 5)
pran.plotPR()
y_pred = pran.predict(X_test)
y_pred = (y_pred>0.5).float()
print("Precision Score: ", precision_score(y_test, y_pred))
print("Recall Score: ", recall_score(y_test, y_pred))