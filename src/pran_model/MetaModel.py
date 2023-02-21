import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class MetaModel(torch.nn.Module):
    '''
    Model for transfer learning
    '''

    inp = 0
    outDim = 1

    def __init__(self, inputDim):
        super(MetaModel, self).__init__()
        self.inp = inputDim
        self.outDim = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(self.inp, 8),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3)
        ) 

        self.hidden1 = nn.Sequential(
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.6)
        ) 

        # self.hidden2 = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.LeakyReLU(0.2),
        #     # nn.Dropout(0.3)
        # )

        self.out = nn.Sequential(
            torch.nn.Linear(8, self.outDim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        # x = self.hidden2(x)
        x = self.out(x)
        return x