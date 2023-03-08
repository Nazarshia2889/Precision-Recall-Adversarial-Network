import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from pran_model.PrecisionBooster import PrecisionBooster
from pran_model.RecallBooster import RecallBooster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data.sampler import WeightedRandomSampler

import random

class PRAN():
    '''
    Precision-Recall Adversarial Network class code
    '''

    precision_booster = None
    recall_booster = None

    precisionTrain = []
    recallTrain = []
    epochsTrain = []

    precisionLoss = []
    recallLoss = []

    '''
    Hyperparameters:
        Input_dim: Number of features in input data
        Mid_dim: Intermediate number of nodes between Recall Booster and Precision Booster
    '''
    def __init__(self, shape=None, mid_dim=16):
        self.recall_booster = RecallBooster(shape, mid_dim)
        self.precision_booster = PrecisionBooster(mid_dim)
        self.precisionTrain = []
        self.recallTrain = []
        self.epochsTrain = []

    '''
    fit() - Train PRAN
    Hyperparameters:
        X_train: Input train data
        y_train: Input train data (labels)
        X_val: Input validation data
        y_train: Input validation data (labels)
        epochs: Number of training iterations
        batch_size: Batch size to input into PRAN at once
        lr_pb: Learning rate for the Precision Booster
        lr_rb: Learning rate for the Recall Booster
        l2_pb: Weight decay (L2) for Precision Booster
        l2_rb: Weight decay (L2) for Recall Booster
        beta_pb: Weight term for 1 class Precision Booster ( < 1)
        beta_rb: Weight term for 1 class Recall Booster ( > 1)
    '''
    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size, lr_pb, lr_rb, l2_pb, l2_rb, weights_pb, weights_rb):
        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.Tensor(y_train)

        my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
        data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
        num_batches = len(data_loader)

        rb_optimizer = optim.Adam(self.recall_booster.parameters(), lr=lr_rb, weight_decay = l2_rb)
        torch.autograd.set_detect_anomaly(True)

        pb_optimizer = optim.Adam(self.precision_booster.parameters(), lr=lr_pb, weight_decay = l2_pb)

        precision_training = False
        recall_training = True

        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                if recall_training:
                    rb_optimizer.zero_grad()
                    output1 = self.recall_booster(batch_x)
                    output1 = self.precision_booster(output1)
                    recall_cost = self.recall_booster.cost(torch.reshape(batch_y, (batch_y.size(0), 1)), torch.reshape(output1, (output1.size(0), 1)), weights = weights_rb)
                    recall_cost.backward()
                    rb_optimizer.step()
                    self.recallLoss.append(recall_cost)
                elif precision_training:
                    output1 = self.recall_booster(batch_x)
                    pb_optimizer.zero_grad()
                    y_pred = self.precision_booster(output1)
                    precision_cost = self.precision_booster.cost(torch.reshape(batch_y, (batch_y.size(0), 1)), torch.reshape(y_pred, (y_pred.size(0), 1)), weights = weights_pb) 
                    precision_cost.backward()
                    pb_optimizer.step()
                    self.precisionLoss.append(precision_cost)
                
                yhat = (self.precision_booster(self.recall_booster(torch.Tensor(X_val))) > 0.5).float()
                precision = self.precision_booster.getPrecision(y_val, yhat)
                recall = self.precision_booster.getRecall(y_val, yhat)
                
                if precision >= recall:
                    precision_training = False
                    recall_training = True
                else:
                    precision_training = True
                    recall_training = False

            self.epochsTrain.append(epoch)
            self.precisionTrain.append(precision)
            self.recallTrain.append(recall)

    '''
    predict() - Input test data and return prediction probabilities
    Hyperparameters:
        X_test: Testing data
    '''
    def predict(self, X_test):
        test_tensor_x = torch.Tensor(X_test)
        output1 = self.recall_booster(test_tensor_x)
        y_pred = self.precision_booster(output1)
        return y_pred
    

    '''
    plotPR() - Plot precision and recall over training
    '''
    def plotPR(self):
        df = pd.DataFrame({'Epochs': self.epochsTrain, 'Precision': self.precisionTrain, 'Recall': self.recallTrain})

        sns.set(style="darkgrid")
        sns.lineplot(x="Epochs", y="Precision", data=df, label="Precision")
        sns.lineplot(x="Epochs", y="Recall", data=df, label="Recall")

        plt.ylim(-0.1, 1.1)
        plt.title("Precision and Recall Over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        
        plt.show()
    
    # def plotCost(self):
    #     df = pd.DataFrame({'Epochs': np.arange(0, len(precisionLoss)), 'Precision': self.precisionLoss})

    #     sns.set(style="darkgrid")
    #     sns.lineplot(x="Epochs", y="Precision", data=df, label="Precision Cost")
    #     sns.lineplot(x="Epochs", y="Recall", data=df, label="Recall")

    #     plt.ylim(-0.1, 1.1)
    #     plt.title("Precision and Recall Over Time")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Value")
        
    #     plt.show()
        