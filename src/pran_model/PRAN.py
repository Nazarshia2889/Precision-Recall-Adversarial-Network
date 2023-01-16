import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score

from PrecisionBooster import PrecisionBooster
from RecallBooster import RecallBooster
from RecallBooster import mseLoss

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

    precisionTrain = []
    recallTrain = []
    epochsTrain = []

    def __init__(self, shape):
        self.precision_booster = PrecisionBooster(shape)
        self.recall_booster = RecallBooster(shape, 50, 12)
    
    def fit(self, X_train, y_train, batch_rb_samples, epochs, batch_size, lr_pb, lr_rb, l2_pb, l2_rb):
        mu, logvar = self.init_train_rb(X_train, y_train, batch_size = 8, lr = 1e-3, epochs = 2000)

        print("--Initial Recall Booster Training Complete--")

        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.Tensor(y_train)

        my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
        data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
        num_batches = len(data_loader)

        pb_optimizer = optim.Adam(self.precision_booster.parameters(), lr=lr_pb, weight_decay = l2_pb)
        rb_optimizer = optim.Adam(self.recall_booster.parameters(), lr=lr_rb, weight_decay = l2_rb)

        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                sigma = torch.exp(logvar/2)
                no_samples = batch_rb_samples
                q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
                z = q.rsample(sample_shape=torch.Size([no_samples]))

                with torch.no_grad():
                    pred = self.recall_booster.decode(z)

                tensor_x_new = torch.vstack((batch_x, pred))
                ones = self.ones_target(no_samples)
                ones = torch.reshape(ones, (ones.size(0), ))
                tensor_y_new = torch.hstack((batch_y, ones))
                
                rb_optimizer.zero_grad()
                pb_optimizer.zero_grad()

                y_pred = self.precision_booster(tensor_x_new).detach()
                y_pred = (y_pred>0.5).float()

                precision = self.precision_booster.getPrecision(tensor_y_new, y_pred)
                recall = self.recall_booster.getRecall(tensor_y_new, y_pred)
                
                if precision >= recall:
                    # recon_batch, mu, logvar = self.recall_booster(batch_x[torch.where(batch_y == 1)])
                    recon_batch, mu, logvar = self.recall_booster(batch_x)
                    recall_cost = self.recall_booster.cost(tensor_y_new, y_pred, mu, logvar)
                    recall_cost.backward()
                    rb_optimizer.step()
                elif precision < recall:
                    precision_cost = self.precision_booster.cost(tensor_y_new, y_pred)
                    precision_cost.backward()
                    pb_optimizer.step()
            self.precisionTrain.append(precision)
            self.recallTrain.append(recall)
            self.epochsTrain.append(epoch)
            print(f"Epoch #{epoch} - \n Precision: {precision} \n Recall: {recall} \n")
    
    def init_train_rb(self, X_train, y_train, batch_size, lr, epochs):
        X_train_pos = X_train[np.where(y_train == 1)[0]]
        y_train_pos = np.where(y_train == 1)[0]

        tensor_x_pos = torch.Tensor(X_train_pos)
        tensor_y_pos = torch.Tensor(y_train_pos)

        train_vae_data = torch.utils.data.TensorDataset(tensor_x_pos, tensor_y_pos)
        train_vae_loader = torch.utils.data.DataLoader(train_vae_data, batch_size=batch_size, shuffle=True)

        init_optim = optim.Adam(self.recall_booster.parameters(), lr = lr)
        loss_mse = mseLoss()

        for epoch in range(epochs):
            self.recall_booster.train()
            for batch_x, batch_y in train_vae_loader:
                init_optim.zero_grad()
                recon_batch, mu, logvar = self.recall_booster(batch_x)
                loss = loss_mse(recon_batch, batch_x, mu, logvar)
                loss.backward(retain_graph=True)
                init_optim.step()

        init_optim.zero_grad()

        return mu, logvar
    
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



