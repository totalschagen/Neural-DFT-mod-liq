import torch.nn as nn
import torch.nn.functional as F



class conv_neural_func7(nn.Module):
    def __init__(self):
        super(conv_neural_func7, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size = 213)
        self.conv2 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv3 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv4 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv5 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv6 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv7 = nn.Conv2d(16,1,kernel_size = 1)

    def forward(self,x):
        x = self.conv1(x)
        x = F.softplus(self.conv2(x))
        x = F.softplus(self.conv3(x))
        x = F.softplus(self.conv4(x))
        x = F.softplus(self.conv5(x))
        x = F.softplus(self.conv6(x))
        x = self.conv7(x)
        return x

class conv_neural_func5(nn.Module):
    def __init__(self):
        super(conv_neural_func5, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size = 85)
        self.conv2 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv3 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv4 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv5 = nn.Conv2d(16,1,kernel_size = 1)

    def forward(self,x):
        x = self.conv1(x)
        x = F.softplus(self.conv2(x))
        x = F.softplus(self.conv3(x))
        x = F.softplus(self.conv4(x))
        x = self.conv5(x)
        return x

class conv_neural_func_new(nn.Module):
    def __init__(self):
        super(conv_neural_func_new, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size = 85)
        self.conv2 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv3 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv4 = nn.Conv2d(16,16,kernel_size = 1)
        self.conv5 = nn.Conv2d(16,1,kernel_size = 1)
        self.conv6 = nn.Conv2d()

    def forward(self,x):
        x = self.conv1(x)
        x = F.softplus(self.conv2(x))
        x = F.softplus(self.conv3(x))
        x = F.softplus(self.conv4(x))
        x = self.conv5(x)
        return x
