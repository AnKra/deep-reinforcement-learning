import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    """Actor (Policy) Model."""

    last_conv_layer_size = 64
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
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, self.last_conv_layer_size, kernel_size=3, stride=1)
        
        #self.maxpool = nn.MaxPool2d(2, 2, padding=0)
        
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_size)
        
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = x.permute(0, 3, 1, 2)
        
        do_print = False
        
        if do_print:
            print(x.shape)
        
        x = F.relu(self.conv1(x))
        #x = self.maxpool(x)
        
        if do_print:
            print(x.shape)
        
        x = F.relu(self.conv2(x))
        #x = self.maxpool(x)
        
        if do_print:
            print(x.shape)

        x = F.relu(self.conv3(x))
        #x = self.maxpool(x)
        
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
        x = F.softmax(self.fc2(x), 1)
        
        if do_print:
            print(x.shape)
            print()
        
        return x

# without maxpool
# last_conv_layer_size = 64
# last_conv_layer_size * 7 * 7
# torch.Size([1, 3, 84, 84])
# torch.Size([1, 32, 20, 20])
# torch.Size([1, 64, 9, 9])
# torch.Size([1, 64, 7, 7])
# torch.Size([1, 3136])
# torch.Size([1, 512])
# torch.Size([1, 4])

# without maxpool
# last_conv_layer_size = 64
# last_conv_layer_size * 7 * 7
# torch.Size([1, 3, 84, 84])
# torch.Size([1, 32, 20, 20])
# torch.Size([1, 64, 9, 9])
# torch.Size([1, 64, 7, 7])
# torch.Size([1, 3136])
# torch.Size([1, 64])
# torch.Size([1, 4])
    
# with maxpool
# last_conv_layer_size = 32
# conv_output_size = last_conv_layer_size * 4 * 4
# torch.Size([1, 3, 84, 84])
# torch.Size([1, 16, 21, 21])
# torch.Size([1, 32, 4, 4])
# torch.Size([1, 512])
# torch.Size([1, 64])
# torch.Size([1, 4])

# without maxpool
# last_conv_layer_size = 32
# conv_output_size = last_conv_layer_size * 20 * 20
# torch.Size([1, 3, 84, 84])
# torch.Size([1, 16, 42, 42])
# torch.Size([1, 32, 20, 20])
# torch.Size([1, 12800])
# torch.Size([1, 64])
# torch.Size([1, 4])

# without maxpool
# last_conv_layer_size = 32
# conv_output_size = last_conv_layer_size * 9 * 9
# torch.Size([1, 3, 84, 84])
# torch.Size([1, 32, 20, 20])
# torch.Size([1, 32, 9, 9])
# torch.Size([1, 2592])
# torch.Size([1, 9])
# torch.Size([1, 4])

# with maxpool
# last_conv_layer_size = 32
# conv_output_size = last_conv_layer_size * 2 * 2
# torch.Size([1, 3, 84, 84])
# torch.Size([1, 32, 10, 10])
# torch.Size([1, 32, 2, 2])
# torch.Size([1, 128])
# torch.Size([1, 9])
# torch.Size([1, 4])


