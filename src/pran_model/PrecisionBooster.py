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
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        # x = self.hidden2(x)
        x = self.out(x)
        return x

    # def cost(self, real, predicted):
    #     precision = precision_score(real, predicted)
    #     # precision_cost = 1 - precision
    #     precision_cost = precision
    #     precision_cost = Variable(torch.tensor(precision_cost, dtype=torch.float32), requires_grad = True)
    #     recall_cost = Variable(torch.tensor(recall_score(real, predicted), dtype=torch.float32), requires_grad = True)
    #     # return (-torch.log(precision_cost + epsilon) * (recall_score(real, predicted) + epsilon)) + (-torch.log(Variable(torch.tensor(recall_score(real, predicted), dtype=torch.float32), requires_grad = True) + epsilon) * (precision_cost + epsilon))
    #     # return ((precision_cost + epsilon) * torch.log(1 + torch.exp(-recall_cost))) + ((recall_cost + epsilon) * torch.log(1 + torch.exp(-precision_cost)))
    #     # return torch.log(1 + torch.exp(-precision_cost))
    #     return 5 * -torch.log(precision_cost + 1e-6)

    def cost(self, real, predicted, beta, epsilon):
        # y_pred = Variable(predicted.float(), requires_grad = True)
        # y_true = Variable(real.float(), requires_grad = True)
        # loss = -(beta * real * torch.log(predicted + epsilon) + (1 - real) * torch.log(1 - predicted + epsilon))
        weights = torch.Tensor([beta])
        bce = nn.BCEWithLogitsLoss(pos_weight=weights)
        loss = bce(predicted, real)
        return loss

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



# import torch
# from torch import nn, optim
# from torch.autograd.variable import Variable
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score

# class PrecisionBooster(torch.nn.Module):
#     '''
#     Model that focuses on precision
#     '''

#     inp = 0
#     outDim = 1

#     def __init__(self, inputDim):
#         super(PrecisionBooster, self).__init__()
#         self.inp = inputDim
#         self.outDim = 1

#         self.hidden0 = nn.Sequential(
#             nn.Linear(self.inp, 16),
#             nn.LeakyReLU(0.2),
#             # nn.Dropout(0.3)
#         ) 

#         self.hidden1 = nn.Sequential(
#             nn.Linear(16, 16),
#             nn.LeakyReLU(0.2),
#             # nn.Dropout(0.6)
#         )
        

#         self.hidden2 = nn.Sequential(
#             nn.Linear(16, 16),
#             nn.LeakyReLU(0.2),
#             # nn.Dropout(0.3)
#         )

#         self.out = nn.Sequential(
#             torch.nn.Linear(16, self.outDim),
#             torch.nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         x = self.hidden0(x)
#         x = self.hidden1(x)
#         x = self.hidden2(x)
#         x = self.out(x)
#         return x

#     def cost(self, real, predicted, epsilon):
#         precision = precision_score(real, predicted)
#         # precision_cost = 1 - precision
#         precision_cost = precision
#         precision_cost = Variable(torch.tensor(precision_cost, dtype=torch.float32), requires_grad = True)
#         recall_cost = Variable(torch.tensor(recall_score(real, predicted), dtype=torch.float32), requires_grad = True)
#         # return (-torch.log(precision_cost + epsilon) * (recall_score(real, predicted) + epsilon)) + (-torch.log(Variable(torch.tensor(recall_score(real, predicted), dtype=torch.float32), requires_grad = True) + epsilon) * (precision_cost + epsilon))
#         return ((precision_cost + epsilon) * torch.log(1 + torch.exp(-recall_cost))) + ((recall_cost + epsilon) * torch.log(1 + torch.exp(-precision_cost)))

#     def getPrecision(self, real, predicted):
#         precision = precision_score(real, predicted)
#         return precision

