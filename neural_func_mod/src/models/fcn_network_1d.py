import torch.nn as nn
import torch.nn.functional as F

class fcn_neural_func(nn.Module):
    def __init__(self):
        super(fcn_neural_func, self).__init__()
        self.fc1 = nn.Linear(213, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 1)

    def forward(self,x):
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = F.softplus(self.fc4(x))
        x = self.fc5(x)
        return x

net = fcn_neural_func()
print(net)