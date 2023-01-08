class PrecisionBooster(torch.nn.Module):
    '''
    Model that focuses on precision
    '''
    def __init__(self, inputDim, hidden_layers=1, units=64):
        super(PrecisionBooster, self).__init__()
        inp = inputDim
        out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(inp, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        ) 

        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            torch.nn.Linear(256, out),
            torch.nn.Sigmoid
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def cost():
        precision = 1
        return 1 - precision
