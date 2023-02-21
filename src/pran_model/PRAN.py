import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from pran_model.PrecisionBooster import PrecisionBooster
from pran_model.RecallBooster import RecallBooster
from pran_model.TransferModel import TransferModel
from pran_model.MetaModel import MetaModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data.sampler import WeightedRandomSampler

import random

class PRAN():
    '''
    Full PRAN model class
    '''

    precision_booster = None
    recall_booster = None

    precisionTrain = []
    recallTrain = []
    epochsTrain = []

    precisionLoss = []
    recallLoss = []

    def __init__(self, shape):
        self.recall_booster = RecallBooster(shape, 16)
        self.precision_booster = PrecisionBooster(16)
        self.transfer_model = TransferModel(shape)
        self.meta_model = MetaModel(2)
        self.precisionTrain = []
        self.recallTrain = []
        self.epochsTrain = []
    
    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size, lr_pb, lr_rb, l2_pb, l2_rb, beta_pb, beta_rb, tolerance):
        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.Tensor(y_train)

        my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
        data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
        num_batches = len(data_loader)

        rb_optimizer = optim.Adam(self.recall_booster.parameters(), lr=lr_rb, weight_decay = l2_rb)
        torch.autograd.set_detect_anomaly(True)

        # for epoch in range(25):
        #     for batch_x, batch_y in data_loader:
        #         rb_optimizer.zero_grad()
        #         output1 = self.recall_booster(batch_x)
        #         # output1 = self.precision_booster(output1)
        #         recall_cost = self.recall_booster.cost(torch.reshape(batch_y, (batch_y.size(0), 1)), torch.reshape(output1, (output1.size(0), 1)), beta = beta_rb, epsilon = 1e-9)
        #         recall_cost.backward()
        #         rb_optimizer.step()
        
        # yhat = (self.precision_booster(self.recall_booster(torch.Tensor(X_val))) > 0.5).float()
        # precision = self.precision_booster.getPrecision(y_val, yhat)
        # recall = self.precision_booster.getRecall(y_val, yhat)
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall} \n")

        pb_optimizer = optim.Adam(self.precision_booster.parameters(), lr=lr_pb, weight_decay = l2_pb)

        # for epoch in range(20):
        #     for batch_x, batch_y in data_loader:
        #         output1 = self.recall_booster(batch_x)
        #         pb_optimizer.zero_grad()
        #         y_pred = self.precision_booster(output1)
        #         precision_cost = self.precision_booster.cost(torch.reshape(batch_y, (batch_y.size(0), 1)), torch.reshape(y_pred, (y_pred.size(0), 1)), beta = beta_pb, epsilon = 1e-9) 
        #         precision_cost.backward()
        #         pb_optimizer.step()

        # yhat = (self.precision_booster(self.recall_booster(torch.Tensor(X_val))) > 0.5).float()
        # precision = self.precision_booster.getPrecision(y_val, yhat)
        # recall = self.precision_booster.getRecall(y_val, yhat)
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall} \n")
        # if precision >= recall:
        #     precision_training = False
        #     recall_training = True
        # else:
        #     precision_training = True
        #     recall_training = False

        precision_training = False
        recall_training = True

        avgPrecisions = []
        avgRecalls = []

        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                if recall_training:
                    rb_optimizer.zero_grad()
                    output1 = self.recall_booster(batch_x)
                    output1 = self.precision_booster(output1)
                    recall_cost = self.recall_booster.cost(torch.reshape(batch_y, (batch_y.size(0), 1)), torch.reshape(output1, (output1.size(0), 1)), beta = beta_rb, epsilon = 1e-9)
                    recall_cost.backward()
                    rb_optimizer.step()
                elif precision_training:
                    output1 = self.recall_booster(batch_x)
                    pb_optimizer.zero_grad()
                    y_pred = self.precision_booster(output1)
                    precision_cost = self.precision_booster.cost(torch.reshape(batch_y, (batch_y.size(0), 1)), torch.reshape(y_pred, (y_pred.size(0), 1)), beta = beta_pb, epsilon = 1e-9) 
                    precision_cost.backward()
                    pb_optimizer.step()
                
                # Added
                
                yhat = (self.precision_booster(self.recall_booster(torch.Tensor(X_val))) > 0.5).float()
                precision = self.precision_booster.getPrecision(y_val, yhat)
                recall = self.precision_booster.getRecall(y_val, yhat)
                
                if precision >= recall:
                    precision_training = False
                    recall_training = True
                else:
                    precision_training = True
                    recall_training = False

            # yhat = (self.precision_booster(self.recall_booster(torch.Tensor(X_val))) > 0.5).float()
            # precision = self.precision_booster.getPrecision(y_val, yhat)
            # recall = self.precision_booster.getRecall(y_val, yhat)
            # avgPrecisions.append(precision)
            # avgRecalls.append(recall)
            # if recall_training:
            #     print("RB Training - ")
            # else:
            #     print("PB Training - ")
            # print(f"Epoch #{epoch}: ")
            # print(f"Precision: {precision}")
            # print(f"Recall: {recall} \n")
            self.epochsTrain.append(epoch)
            self.precisionTrain.append(precision)
            self.recallTrain.append(recall)

            # if epoch % tolerance == 0:
            #     if np.mean(avgPrecisions) >= np.mean(avgRecalls):
            #         precision_training = False
            #         recall_training = True
            #     else:
            #         precision_training = True
            #         recall_training = False
            #     avgPrecisions = []
            #     avgRecalls = []

            # recall = self.recall_booster.getRecall(batch_y, (output1 > 0.5).float())
            # precision_test = self.recall_booster.getPrecision(batch_y, (output1 > 0.5).float())
            # self.precisionTrain.append(precision_test)
            # self.recallTrain.append(recall)
            # self.epochsTrain.append(epoch)
            # self.precisionLoss.append(self.precision_booster.cost(batch_y, output2, beta = 2, epsilon = 1e-9).item())
            # self.recallLoss.append(self.recall_booster.cost(batch_y, output1, beta = 1e-1, epsilon = 1e-9).item())
            # print(f"Epoch #{epoch}, RB - \n Precision: {precision_test} \n Recall: {recall}\n")
            # print("Num pos samples: ", len(torch.where(batch_y == 1)[0]), "\n\n")
        
        # test_tensor_x = torch.Tensor(X_test)
        # output = self.recall_booster(test_tensor_x).detach()
        # print("Precision Score: ", self.recall_booster.getPrecision(y_test, (output > 0.5).float()))
        # print("Recall Score: ", self.recall_booster.getRecall(y_test, (output > 0.5).float()))

        # print('Recall booster training done')
        # print('\n')

        # for epoch in range(epochs):
        #     for batch_x, batch_y in data_loader:

        #     precision = self.precision_booster.getPrecision(batch_y, (y_pred > 0.5).float())
        #     recall_test = self.precision_booster.getRecall(batch_y, (y_pred > 0.5).float())
            # print(f"Epoch #{epoch}, PB - \n Precision: {precision} \n Recall: {recall_test}\n")
            # print("Num pos samples: ", len(torch.where(batch_y == 1)[0]), "\n\n")
    
    def predict(self, X_test):
        test_tensor_x = torch.Tensor(X_test)
        output1 = self.recall_booster(test_tensor_x)
        y_pred = self.precision_booster(output1)
        return y_pred
    
    def ones_target(self, size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = Variable(torch.ones(size, 1))
        return data
    
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





# class PRAN():
#     '''
#     Full PRAN model class
#     '''

#     precision_booster = None
#     recall_booster = None

#     precisionTrain = []
#     recallTrain = []
#     precisionTrain2 = []
#     recallTrain2 = []
#     epochsTrain = []

#     precisionLoss = []
#     recallLoss = []

#     def __init__(self, shape):
#         self.recall_booster = RecallBooster(shape)
#         self.precision_booster = PrecisionBooster(1)
#         self.transfer_model = TransferModel(shape)
#         self.meta_model = MetaModel(2)
    
#     def fit(self, X_train, y_train, epochs, batch_size, lr_pb, lr_rb, l2_pb, l2_rb, beta_pb, beta_rb, X_test, y_test):
#         tensor_x = torch.Tensor(X_train)
#         tensor_y = torch.Tensor(y_train)

#         my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
#         data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size)
#         num_batches = len(data_loader)

#         rb_optimizer = optim.Adam(self.recall_booster.parameters(), lr=lr_rb, weight_decay = l2_rb)

#         torch.autograd.set_detect_anomaly(True)

#         for epoch in range(epochs):
#             for batch_x, batch_y in data_loader:
#                 rb_optimizer.zero_grad()
#                 output1 = self.recall_booster(batch_x)
#                 recall_cost = self.recall_booster.cost(torch.reshape(batch_y, (batch_y.size(0), 1)), torch.reshape(output1, (output1.size(0), 1)), beta = beta_rb, epsilon = 1e-9)
#                 recall_cost.backward()
#                 rb_optimizer.step()

#             recall = self.recall_booster.getRecall(batch_y, (output1 > 0.5).float())
#             precision_test = self.recall_booster.getPrecision(batch_y, (output1 > 0.5).float())
#             self.precisionTrain.append(precision_test)
#             self.recallTrain.append(recall)
#             self.epochsTrain.append(epoch)
#             # self.precisionLoss.append(self.precision_booster.cost(batch_y, output2, beta = 2, epsilon = 1e-9).item())
#             # self.recallLoss.append(self.recall_booster.cost(batch_y, output1, beta = 1e-1, epsilon = 1e-9).item())
#             print(output1)
#             print(f"Epoch #{epoch}, RB - \n Precision: {precision_test} \n Recall: {recall}\n")
#             print("Num pos samples: ", len(torch.where(batch_y == 1)[0]), "\n\n")
            
        
#         test_tensor_x = torch.Tensor(X_test)
#         output = self.recall_booster(test_tensor_x).detach()
#         print("Precision Score: ", self.recall_booster.getPrecision(y_test, (output > 0.5).float()))
#         print("Recall Score: ", self.recall_booster.getRecall(y_test, (output > 0.5).float()))

#         print('Recall booster training done')
#         print('\n')

#         pb_optimizer = optim.Adam(self.precision_booster.parameters(), lr=lr_pb, weight_decay = l2_pb)

#         for epoch in range(epochs):
#             for batch_x, batch_y in data_loader:
#                 output1 = self.recall_booster(batch_x)
#                 pb_optimizer.zero_grad()
#                 y_pred = self.precision_booster(output1)
#                 precision_cost = self.precision_booster.cost(torch.reshape(batch_y, (batch_y.size(0), 1)), torch.reshape(y_pred, (y_pred.size(0), 1)), beta = beta_pb, epsilon = 1e-9) 
#                 precision_cost.backward()
#                 pb_optimizer.step()

#             precision = self.precision_booster.getPrecision(batch_y, (y_pred > 0.5).float())
#             recall_test = self.precision_booster.getRecall(batch_y, (y_pred > 0.5).float())
#             self.precisionTrain2.append(precision)
#             self.recallTrain2.append(recall_test)
#             print(y_pred)
#             print(f"Epoch #{epoch}, PB - \n Precision: {precision} \n Recall: {recall_test}\n")
#             print("Num pos samples: ", len(torch.where(batch_y == 1)[0]), "\n\n")
    
#     def predict(self, X_test):
#         test_tensor_x = torch.Tensor(X_test)
#         output1 = self.recall_booster(test_tensor_x)
#         y_pred = self.precision_booster(output1)
#         return y_pred
    
#     def ones_target(self, size):
#         '''
#         Tensor containing ones, with shape = size
#         '''
#         data = Variable(torch.ones(size, 1))
#         return data
    
#     def plotPR(self):
#         sns.set(style="darkgrid")
#         my_dict = {"Precision": self.precisionTrain, "Recall": self.recallTrain, "Epochs": self.epochsTrain}
#         my_dict2 = {"Precision": self.precisionTrain2, "Recall": self.recallTrain2, "Epochs": self.epochsTrain}
#         data = pd.DataFrame(my_dict)
#         data2 = pd.DataFrame(my_dict2)
#         fig, ax = plt.subplots()
#         ax = sns.lineplot(x='Epochs', y='Precision', data=data)
#         ax = sns.lineplot(x='Epochs', y='Recall', data=data)
#         ax = sns.lineplot(x='Epochs', y='Precision', data=data2)
#         ax = sns.lineplot(x='Epochs', y='Recall', data=data2)
#         ax.legend(['Precision (RB)', 'Recall (RB)', 'Precision (PB)', 'Recall (PB)'])
#         plt.title("PR Training")
#         plt.xlabel("Epochs")
#         plt.ylabel("Value")
#         plt.show()

# class PRAN():
#     '''
#     Full PRAN model class
#     '''

#     precision_booster = None
#     recall_booster = None

#     precisionTrain = []
#     recallTrain = []
#     epochsTrain = []

#     precisionLoss = []
#     recallLoss = []

#     def __init__(self, shape):
#         self.recall_booster = RecallBooster(shape)
#         self.precision_booster = PrecisionBooster(shape)
#         self.transfer_model = TransferModel(shape)
#         self.meta_model = MetaModel(2)
    
#     def fit(self, X_train, y_train, epochs, batch_size, lr_pb, lr_rb, l2_pb, l2_rb, beta_pb, beta_rb):
#         tensor_x = torch.Tensor(X_train)
#         tensor_y = torch.Tensor(y_train)

#         self.transfer_model.fit(tensor_x, tensor_y, epochs=50, lr=1e-3, batch_size=8)

#         self.recall_booster.setWeights(self.transfer_model.getWeights())
#         self.recall_booster.setBiases(self.transfer_model.getBiases())

#         self.precision_booster.setWeights(self.transfer_model.getWeights())
#         self.precision_booster.setBiases(self.transfer_model.getBiases())

#         self.precision_booster.setWeights(self.transfer_model.getWeights())
#         self.precision_booster.setBiases(self.transfer_model.getBiases())

#         my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
#         data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle = True)
#         num_batches = len(data_loader)

#         rb_optimizer = optim.Adam(self.recall_booster.parameters(), lr=lr_rb, weight_decay = l2_rb)
#         pb_optimizer = optim.Adam(self.precision_booster.parameters(), lr=lr_pb, weight_decay = l2_pb)

#         torch.autograd.set_detect_anomaly(True)

#         for epoch in range(epochs):
#             for batch_x, batch_y in data_loader:
#                 rb_optimizer.zero_grad()
#                 pb_optimizer.zero_grad()

#                 output1 = self.recall_booster(batch_x).detach()
#                 # output1 = (output1 > 0.5).float()
#                 recall_cost = self.recall_booster.cost(batch_y, output1, beta = beta_rb, epsilon = 1e-9)
#                 recall_cost.backward()
#                 rb_optimizer.step()

#                 output2 = self.precision_booster(batch_x).detach()
#                 # output2 = (output2 > 0.5).float()
#                 precision_cost = self.precision_booster.cost(batch_y, output2, beta = beta_pb, epsilon = 1e-9)
#                 precision_cost.backward()
#                 pb_optimizer.step()

#                 precision = self.precision_booster.getPrecision(batch_y, (output2 > 0.5).float())
#                 recall_test = self.precision_booster.getRecall(batch_y, (output2 > 0.5).float())

#                 recall = self.recall_booster.getRecall(batch_y, (output1 > 0.5).float())
#                 precision_test = self.recall_booster.getPrecision(batch_y, (output1 > 0.5).float())

#                 # if precision > recall:
#                 # else
#             self.precisionTrain.append(precision)
#             self.recallTrain.append(recall)
#             self.epochsTrain.append(epoch)
#             self.precisionLoss.append(self.precision_booster.cost(batch_y, output2, beta = 2, epsilon = 1e-9).item())
#             self.recallLoss.append(self.recall_booster.cost(batch_y, output1, beta = 1e-1, epsilon = 1e-9).item())
#             print(f"Epoch #{epoch}, PB - \n Precision: {precision} \n Recall: {recall_test} \n")
#             print(f"Epoch #{epoch}, RB - \n Precision: {precision_test} \n Recall: {recall} \n\n")
        
#         optimizer = optim.Adam(self.meta_model.parameters(), lr=1e-3, weight_decay = 1e-4)
#         bce = torch.nn.BCELoss()

#         for epoch in range(epochs):
#             for batch_x, batch_y in data_loader:
#                 output1 = self.recall_booster(batch_x)
#                 output2 = self.precision_booster(batch_x)
#                 outputs = torch.cat([output1, output2], 1)

#                 optimizer.zero_grad()
#                 y_pred = self.meta_model(outputs)
#                 print(y_pred)
#                 loss = bce(torch.reshape(y_pred, (y_pred.size(0), 1)), torch.reshape(batch_y, (batch_y.size(0), 1)))
#                 loss.backward()
#                 optimizer.step()
    
#     def predict(self, X_test):
#         test_tensor_x = torch.Tensor(X_test)
#         output1 = self.recall_booster(test_tensor_x)
#         output2 = self.precision_booster(test_tensor_x)
#         outputs = torch.cat([output1, output2], 1)
#         y_pred = self.meta_model(outputs)
#         return y_pred
    
#     def ones_target(self, size):
#         '''
#         Tensor containing ones, with shape = size
#         '''
#         data = Variable(torch.ones(size, 1))
#         return data
    
#     def plotPR(self):
#         sns.set(style="darkgrid")
#         my_dict = {"Precision": self.precisionTrain, "Recall": self.recallTrain, "Epochs": self.epochsTrain}
#         data = pd.DataFrame(my_dict)
#         fig, ax = plt.subplots()
#         ax = sns.lineplot(x='Epochs', y='Precision', data=data)
#         ax1 = sns.lineplot(x='Epochs', y='Recall', data=data)
#         ax.legend(['Precision', 'Recall'])
#         plt.title("PR Training")
#         plt.xlabel("Epochs")
#         plt.ylabel("Value")
#         plt.show()

#         print(self.precisionLoss)
#         print(self.recallLoss)



