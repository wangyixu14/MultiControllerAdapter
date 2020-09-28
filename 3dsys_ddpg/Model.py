# Define the Neural Network architecture used for controllers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # if fc2_units:
        #     self.fc1 = nn.Linear(state_size, fc1_units)
        #     self.fc2 = nn.Linear(fc1_units, fc2_units)
        #     self.fc3 = nn.Linear(fc2_units, action_size)
        # else:
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc3 = nn.Linear(fc1_units, action_size)            
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #     # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        # if self.fc2_units:
        #     x = F.relu(self.fc1(x))
        #     x = F.relu(self.fc2(x))
        #     x = self.fc3(x)
        # else:
        x = F.relu(self.fc1(x))
        x = self.fc3(x)            
        return torch.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=100, fc2_units=100):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fc1(state)
        xs = self.bn1(xs)
        xs = F.leaky_relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        return self.fc3(x)


class IndividualModel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=50):
        super(IndividualModel, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        self.seed = torch.manual_seed(seed)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# class Individualdistill(nn.Module):
#     def __init__(self, state_size, action_size, seed, fc1_units=50):
#         super(Individualdistill, self).__init__()
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, action_size)
#         self.seed = torch.manual_seed(seed)

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

if __name__ == '__main__':
    from thop import profile
    net = Actor(2, 1, 0).to('cuda')

    input_size = 2
    input = torch.randn(1, input_size).to('cuda')
    print(input)
    action = torch.rand(1, 1).to('cuda')
    flops, params = profile(net, inputs=(input, ))
    print(net(input))
# #print('flops: {}, params: {}'.format(flops, params))
    print('flops: {}, params: {}'.format(flops, params))