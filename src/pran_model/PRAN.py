import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score

from PrecisionBooster import PrecisionBooster
from RecallBooster import RecallBooster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PRAN():
    '''
    Full PRAN model class
    '''

    precision_booster = None
    recall_booster = None

    X_train = None
    y_train = None

    rb_samples = 0

    precisionTrain = []
    recallTrain = []
    epochsTrain = []

    def __init__(self, X_train, y_train, rb_samples=100):
        self.X_train = X_train
        self.y_train = y_train

        self.rb_samples = rb_samples

        self.precision_booster = PrecisionBooster(X_train.shape[1])
        self.recall_booster = RecallBooster(X_train.shape[1], self.rb_samples)
    
    def fit(self, epochs, batch_size):
        tensor_x = torch.Tensor(self.X_train)
        tensor_y = torch.Tensor(self.y_train)

        my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
        data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
        num_batches = len(data_loader)

        pb_optimizer = optim.Adam(self.precision_booster.parameters(), lr=0.001)
        rb_optimizer = optim.Adam(self.recall_booster.parameters(), lr=0.001)

        self.epochsTrain = np.arange(0, epochs)

        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:

                generated_data = self.recall_booster(batch_x)
                generated_data = torch.reshape(generated_data, (batch_x.size(0), self.X_train.shape[1]))
                tensor_x_new = torch.vstack((batch_x, generated_data))
                ones = self.ones_target(generated_data.size(0))
                ones = torch.reshape(ones, (ones.size(0), ))
                tensor_y_new = torch.hstack((batch_y, ones))

                y_pred = self.precision_booster(tensor_x_new).detach()
                y_pred = (y_pred>0.5).float()

                precision = self.precision_booster.getPrecision(tensor_y_new, y_pred)
                recall = self.recall_booster.getRecall(tensor_y_new, y_pred)
                
                rb_optimizer.zero_grad()
                pb_optimizer.zero_grad()

                if precision >= recall:
                    recall_cost = self.recall_booster.cost(tensor_y_new, y_pred)
                    recall_cost.backward()
                    rb_optimizer.step()
                elif precision < recall:
                    precision_cost = self.precision_booster.cost(tensor_y_new, y_pred)
                    precision_cost.backward()
                    pb_optimizer.step()
            self.precisionTrain.append(precision)
            self.recallTrain.append(recall)
    
    def predict(self, X_test):
        test_tensor_x = torch.Tensor(X_test)
        y_pred = self.precision_booster(test_tensor_x)
        return y_pred
    
    def ones_target(self, size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = Variable(torch.ones(size, 1))
        return data
    
    def plotPR(self):
        sns.set(style="darkgrid")
        my_dict = {"Precision": self.precisionTrain, "Recall": self.recallTrain, "Epochs": self.epochsTrain}
        data = pd.DataFrame(my_dict)
        fig, ax = plt.subplots()
        ax = sns.lineplot(x='Epochs', y='Precision', data=data)
        ax1 = sns.lineplot(x='Epochs', y='Recall', data=data)
        ax.legend(['Precision', 'Recall'])
        plt.title("PR Training")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.show()



