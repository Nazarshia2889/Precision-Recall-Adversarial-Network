import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score

class PrecisionBooster(torch.nn.Module):
    '''
    Model that focuses on precision
    '''

    inp = 0
    outDim = 1

    def __init__(self, inputDim):
        super(PrecisionBooster, self).__init__()
        self.inp = inputDim
        self.outDim = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(self.inp, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        ) 

        self.hidden1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            torch.nn.Linear(32, self.outDim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def cost(self, real, predicted):
        precision = precision_score(real, predicted)
        precision_cost = 1 - precision
        return Variable(torch.tensor(precision_cost, dtype=torch.float32), requires_grad = True)

    def getPrecision(self, real, predicted):
        precision = precision_score(real, predicted)
        return precision

