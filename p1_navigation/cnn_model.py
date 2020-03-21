import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CNNNetwork(nn.Module):
    """Actor (Policy) Model."""

    last_conv_layer_size = 16
    conv_output_size = last_conv_layer_size * 7 * 7
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(CNNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(2, 4), stride=(1, 2))                 # 42x84 --> 40x40
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)                          # 40x40 --> 18x18
        self.conv3 = nn.Conv2d(16, self.last_conv_layer_size, kernel_size=4, stride=2)  # 18x18 --> 7x7
        
        self.bn1=nn.BatchNorm2d(8)
        self.bn2=nn.BatchNorm2d(16)
        self.bn3=nn.BatchNorm2d(self.last_conv_layer_size)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_size)
        
        self.reset_parameters()
        
        self.dropout = nn.Dropout(p=0.4)
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = x.permute(0, 3, 1, 2)
        
        do_print = False
        
        if do_print:
            print(x.shape)
        
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool(x)
        
        if do_print:
            print(x.shape)
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        
        if do_print:
            print(x.shape)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.maxpool(x)
        
        if do_print:
            print(x.shape)
        
        x = x.view(-1, self.conv_output_size)
        
        if do_print:
            print(x.shape)
        
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        if do_print:
            print(x.shape)
        
        #x = self.dropout(x)
        x = self.fc2(x)
        
        if do_print:
            print(x.shape)
            print()
        
        return x
