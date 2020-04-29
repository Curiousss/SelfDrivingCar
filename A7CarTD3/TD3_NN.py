import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ## Step 2: We build one neural network for the Actor model and one neural network for the Actor target

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.BN3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.BN4 = nn.BatchNorm2d(256)

        self.layer_1 = nn.Linear(256 + 7, 400)
        self.BN5 = nn.BatchNorm1d(400)
        self.layer_2 = nn.Linear(400, 300)
        self.BN6 = nn.BatchNorm1d(300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):

        o = F.max_pool2d(F.relu((self.conv1(x[0]))), 2)
        o = self.BN1(o)
        o = F.max_pool2d(F.relu((self.conv2(o))), 2)
        o = self.BN2(o)
        o = F.max_pool2d(F.relu((self.conv3(o))), 2)
        o = self.BN3(o)
        o = F.max_pool2d(F.relu((self.conv4(o))), 2)
        o = self.BN4(o)
        o = F.avg_pool2d(o, kernel_size=o.shape[-2])
        o = o.view(-1, 256)

        o = torch.cat([o, x[1]], 1)

        o = F.relu(self.layer_1(o))
        o = self.BN5(o)
        o = F.relu(self.layer_2(o))
        o = self.BN6(o)
        o = self.max_action * torch.tanh(self.layer_3(o))
        return o


# ## Step 3: We build two neural networks for the two Critic models and two neural networks for the two Critic targets

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        # -----------------------------------CRITIC 1-----------
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.BN3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.BN4 = nn.BatchNorm2d(256)

        self.layer_5 = nn.Linear(7+256+ + action_dim, 400)
        self.BN5 = nn.BatchNorm1d(400)
        self.layer_6 = nn.Linear(400, 300)
        self.BN6 = nn.BatchNorm1d(300)
        self.layer_7 = nn.Linear(300, 1)

        # -----------------------------------CRITIC 2-----------
        self.conv8 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.BN8 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.BN9 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.BN10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.BN11 = nn.BatchNorm2d(256)
        # Defining the second Critic neural network

        self.layer_12 = nn.Linear(256+7 + action_dim, 400)
        self.BN12 = nn.BatchNorm1d(400)
        self.layer_13 = nn.Linear(400, 300)
        self.BN13 = nn.BatchNorm1d(300)
        self.layer_14 = nn.Linear(300, 1)

    def forward(self, x, u):
        # Forward-Propagation on the first Critic Neural Network
        #xu = torch.Tensor(u).to(device)
        #-----------------------------------CRITIC 1-----------
        x1 = F.max_pool2d(F.relu((self.conv1(x[0]))), 2)
        x1 = self.BN1(x1)
        x1 = F.max_pool2d(F.relu((self.conv2(x1))), 2)
        x1 = self.BN2(x1)
        x1 = F.max_pool2d(F.relu((self.conv3(x1))), 2)
        x1 = self.BN3(x1)
        x1 = F.max_pool2d(F.relu((self.conv4(x1))), 2)
        x1 = self.BN4(x1)
        x1 = F.avg_pool2d(x1, kernel_size=x1.shape[-2])
        x1 = x1.view(-1, 256)

        xu1 = torch.cat([x1, x[1]], 1)
        xu1 = torch.cat([xu1, u], 1)

        x1 = F.relu(self.layer_5(xu1))
        x1 = self.BN5(x1)
        x1 = F.relu(self.layer_6(x1))
        x1 = self.BN6(x1)
        x1 = self.layer_7(x1)

        # Forward-Propagation on the second Critic Neural Network
        # -----------------------------------CRITIC 2-----------
        x2 = F.max_pool2d(F.relu((self.conv8(x[0]))), 2)
        x2 = self.BN8(x2)
        x2 = F.max_pool2d(F.relu((self.conv9(x2))), 2)
        x2 = self.BN9(x2)
        x2 = F.max_pool2d(F.relu((self.conv10(x2))), 2)
        x2 = self.BN10(x2)
        x2 = F.max_pool2d(F.relu((self.conv11(x2))), 2)
        x2 = self.BN11(x2)
        x2 = F.avg_pool2d(x2, kernel_size=x2.shape[-2])
        x2 = x2.view(-1, 256)

        xu2 = torch.cat([x2, x[1]], 1)
        xu2 = torch.cat([xu2, u], 1)

        x2 = F.relu(self.layer_12(xu2))
        x2 = self.BN12(x2)
        x2 = F.relu(self.layer_13(x2))
        x2 = self.BN13(x2)
        x2 = self.layer_14(x2)

        return x1, x2

    def Q1(self, x, u):
        #xu = torch.Tensor(u).to(device)

        x1 = F.max_pool2d(F.relu((self.conv1(x[0]))), 2)
        x1 = self.BN1(x1)
        x1 = F.max_pool2d(F.relu((self.conv2(x1))), 2)
        x1 = self.BN2(x1)
        x1 = F.max_pool2d(F.relu((self.conv3(x1))), 2)
        x1 = self.BN3(x1)
        x1 = F.max_pool2d(F.relu((self.conv4(x1))), 2)
        x1 = self.BN4(x1)
        x1 = F.avg_pool2d(x1, kernel_size=x1.shape[-2])
        x1 = x1.view(-1, 256)

        xu1 = torch.cat([x1,x[1], u], 1)

        x1 = F.relu(self.layer_5(xu1))
        x1 = self.BN5(x1)
        x1 = F.relu(self.layer_6(x1))
        x1 = self.BN6(x1)
        x1 = self.layer_7(x1)
        return x1