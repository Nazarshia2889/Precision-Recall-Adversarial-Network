import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
            nn.Linear(self.inp, 16)
            # nn.Dropout(0.3)
        ) 

        self.hidden1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2)
            # nn.Dropout(0.3)
        )
        
        self.out = nn.Sequential(
            torch.nn.Linear(16, self.outDim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)
        return x

    # def cost(self, real, predicted, beta, epsilon):
    #     weights = torch.Tensor([beta])
    #     bce = nn.BCEWithLogitsLoss(pos_weight=weights)
    #     loss = bce(predicted, real)
    #     return loss
    def cost(self, y_true, y_pred, weights):
        y_pred = torch.clamp(y_pred,min=1e-7,max=1-1e-7)
        bce = - weights[1] * y_true * torch.log(y_pred) - (1 - y_true) * weights[0] * torch.log(1 - y_pred)
        return torch.mean(bce)

    def getPrecision(self, real, predicted):
        precision = precision_score(real, predicted)
        return precision
    
    def getRecall(self, real, predicted):
        recall = recall_score(real, predicted)
        return recall
    
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