import torch
from torch import nn, optim
import torch.nn.functional as F 
from torch import nn, optim 
from torch.autograd import Variable 

import pandas as pd 
import numpy as np 
from sklearn import preprocessing 

from sklearn.metrics import recall_score

class RecallBooster(nn.Module):
    '''
    Model that focuses on recall
    '''
    def __init__(self,D_in,H=50,H2=12,latent_dim=3):
        
        #Encoder
        super(RecallBooster,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
        # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def cost(self, real, predicted, mu, logvar):
        recall = recall_score(real, predicted)
        recall_cost = 1 - recall
        recall_cost = Variable(torch.tensor(recall_cost, dtype=torch.float32), requires_grad = True)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # return Variable(torch.tensor(recall_cost, dtype=torch.float32), requires_grad = True)
        return -torch.log(recall_cost) + loss_KLD
    
    def getRecall(self, real, predicted):
        recall = recall_score(real, predicted)
        return recall

class mseLoss(nn.Module):
    def __init__(self):
        super(mseLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

# import torch
# from torch import nn, optim
# from torch.autograd.variable import Variable
# from sklearn.metrics import recall_score

# class RecallBooster(torch.nn.Module):
#     '''
#     Model that focuses on recall
#     '''

#     inp = 0
#     outDim = 1

#     def __init__(self, inputDim, numSamples):
#         super(RecallBooster, self).__init__()
#         self.inp = inputDim
#         self.outDim = inputDim # * numSamples

#         self.hidden0 = nn.Sequential(
#             nn.Linear(self.inp, 256),
#             nn.LeakyReLU(0.2),
#         ) 

#         self.hidden1 = nn.Sequential(
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#         )

#         self.hidden2 = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3)
#         )

#         self.out = nn.Sequential(
#             torch.nn.Linear(1024, self.outDim),
#             torch.nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         x = self.hidden0(x)
#         x = self.hidden1(x)
#         x = self.hidden2(x)
#         x = self.out(x)
#         return x

#     def cost(self, real, predicted):
#         recall = recall_score(real, predicted)
#         recall_cost = 1 - recall
#         return Variable(torch.tensor(recall_cost, dtype=torch.float32), requires_grad = True)
    
#     def getRecall(self, real, predicted):
#         recall = recall_score(real, predicted)
#         return recall
    

