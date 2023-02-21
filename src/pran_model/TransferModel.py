import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TransferModel(torch.nn.Module):
    '''
    Model for transfer learning
    '''

    inp = 0
    outDim = 1

    precisionTrain = []
    recallTrain = []
    epochsTrain = []

    def __init__(self, inputDim):
        super(TransferModel, self).__init__()
        self.inp = inputDim
        self.outDim = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(self.inp, 16),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3)
        ) 

        self.hidden1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.6)
        ) 

        # self.hidden2 = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.LeakyReLU(0.2),
        #     # nn.Dropout(0.3)
        # )

        self.out = nn.Sequential(
            torch.nn.Linear(16, self.outDim)
            # torch.nn.Sigmoid()
        )

        self.precisionTrain = []
        self.recallTrain = []
        self.epochsTrain = []
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        # x = self.hidden2(x)
        x = self.out(x)
        return x

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size):
        optimizer = optim.Adam(self.parameters(), lr = lr)
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train)

        my_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size)

        bce = nn.BCEWithLogitsLoss()
        epoch = 0
        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                y_pred = self(batch_x)
                loss = bce(torch.reshape(y_pred, (y_pred.size(0), 1)), torch.reshape(batch_y, (batch_y.size(0), 1)))
                loss.backward()
                optimizer.step()
            yhat = (self(torch.Tensor(X_val)) > 0.5).float()
            precision = precision_score(y_val, yhat)
            recall = recall_score(y_val, yhat)
            self.epochsTrain.append(epoch)
            self.precisionTrain.append(precision)
            self.recallTrain.append(recall)

            epoch += 1
    
    def evaluate(self, X_test, y_test):
        X_test = torch.Tensor(X_test)
        y_pred = self(X_test)
        y_pred = (y_pred > 0.5).float()
        return precision_score(y_test, y_pred), recall_score(y_test, y_pred)
    
    def plotPR(self):
        # create a pandas dataframe
        df = pd.DataFrame({'Epochs': self.epochsTrain, 'Precision': self.precisionTrain, 'Recall': self.recallTrain})

        # create the line plot using Seaborn
        sns.set(style="darkgrid")
        sns.lineplot(x="Epochs", y="Precision", data=df, label="Precision")
        sns.lineplot(x="Epochs", y="Recall", data=df, label="Recall")

        plt.ylim(-0.1, 1.1)

        # set plot title and axis labels
        plt.title("Precision and Recall Over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Value")

        # show the plot
        plt.show()

    def getWeights(self):
        return self.hidden0[0].weight.data

    def getBiases(self):
        return self.hidden0[0].bias.data
    
    def setWeights(self, layer_data):
        self.hidden0[0].weight.data = layer_data
        self.hidden0[0].weight.requiresGrad = False

    def setBiases(self, layer_data):
        self.hidden0[0].bias.data = layer_data
        self.hidden0[0].bias.requiresGrad = False