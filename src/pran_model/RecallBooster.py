import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import recall_score

class RecallBooster(torch.nn.Module):
    '''
    Model that focuses on recall
    '''

    inp = 0
    outDim = 1

    def __init__(self, inputDim, numSamples):
        super(RecallBooster, self).__init__()
        self.inp = inputDim
        self.outDim = inputDim # * numSamples

        self.hidden0 = nn.Sequential(
            nn.Linear(self.inp, 256),
            nn.LeakyReLU(0.2),
        ) 

        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            torch.nn.Linear(1024, self.outDim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def cost(self, real, predicted):
        recall = recall_score(real, predicted)
        recall_cost = 1 - recall
        return Variable(torch.tensor(recall_cost, dtype=torch.float32), requires_grad = True)
    
    def getRecall(self, real, predicted):
        recall = recall_score(real, predicted)
        return recall
    

