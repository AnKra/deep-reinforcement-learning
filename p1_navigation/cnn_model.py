import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    """Actor (Policy) Model."""

    last_conv_layer_size = 32
    conv_output_size = last_conv_layer_size * 2 * 2
    
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
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, self.last_conv_layer_size, kernel_size=4, stride=2)
        
        self.maxpool = nn.MaxPool2d(2, 2, padding=0)
        
        self.fc1 = nn.Linear(self.conv_output_size, 9)
        self.fc2 = nn.Linear(9, action_size)
        
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        
        x = x.view(-1, self.conv_output_size)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
