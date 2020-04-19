# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Importing the libraries
import torch
from torch.autograd import Variable
# Creating the architecture of the Neural Network
# Selecting the device (CPU or GPU)

from PIL import Image as PILImage
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MEERA
class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        return x

class MNIST_FeatureExtractor:

    #MEERA
    def method_feature_extration(state, iS=(4,100, 100), model_name = 'MNIST_model.pth'):

        model = MNIST_Net()
        state_dict = torch.load(model_name) #"MNIST_model.pth")
        model.load_state_dict(state_dict)
        model.eval()

        car_features_frames = []
        for i in range(iS[0]):

            pixel_state = state[:, i, :, :]
            pixel_state = np.reshape(pixel_state, (pixel_state.shape[1], pixel_state.shape[2]))
            pixel_state_img = PILImage.fromarray(np.uint8(pixel_state * 255))

            preprocess = transforms.Compose(
                [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            car_tensor = preprocess(pixel_state_img).float()
            car_tensor = car_tensor.unsqueeze_(0)
            car_features = model(car_tensor)
            car_features_frames.append(car_features)

        car_features_vector = torch.cat((car_features_frames[0], car_features_frames[1], car_features_frames[2], car_features_frames[3]))

        return np.array(car_features_vector.data)



if __name__ == '__main__':

    img = PILImage.open("./images/mask.png").convert('L')
    # img = PILImage.open("./images/mask.png")
    temp_sand = np.asarray(img) / 255

    #Car's center position
    xx = 100
    yy = 100

    # crop area 100x100
    nLastStates = 4
    rows = 100
    columns = 100

    pixel_state = temp_sand [int(xx) - rows//2:int(xx) + rows//2, int(yy) - columns//2:int(yy) + columns//2] #a 100x100 2 D gray scale crop area

    pil_img = PILImage.fromarray(np.uint8(pixel_state * 255))

    #readying the inputs to CNN
    currentState = np.zeros((1, nLastStates, rows, columns))  # currentState = np.zeros((1, nLastStates, rows, columns))
    for i in range(nLastStates):
        currentState[:, i, :, :] = pixel_state  # a rows x columns numpy array

    #DCQN_FeatureExtractor.method_feature_extration(state=currentState, iS=(4,100, 100), nb_action = 3, model_name = 'last_brain_54000.pth' )

    temp_tensor = MNIST_FeatureExtractor.method_feature_extration(currentState, iS=(4,100, 100), model_name = 'model.pth')
    print(type(temp_tensor))
    print(temp_tensor.size())

    # Sess10:  <class 'generator'>
    # <class 'torch.Tensor'>
    # torch.Size([1, 64, 93, 93])
    # torch.Size([1, 64, 93, 93])
    # torch.Size([553536])