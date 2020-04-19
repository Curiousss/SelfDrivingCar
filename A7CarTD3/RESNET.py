#import torchvision
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
from PIL import Image as PILImage

#class MNIST_FeatureExtractor:

    # MEERA
def getstate_image(x, y, car_angle, car_size):
    #map_np = np.ones(map_size)  # (548, 975)
    #sand = np.zeros((975, 548))
    map_img = PILImage.open("./images/mask.png")
    # img = PILImage.open("./images/mask.png")
    #sand = np.asarray(img) / 255
    #map_img = PILImage.fromarray(sand) #map_np.astype("uint8") * 255)
    #map_img.save('blank.png')
    arrow_img = PILImage.open('arrow.png')
    arrow_img = arrow_img.rotate(car_angle)
    arrow_img = arrow_img.resize(car_size)
    print(arrow_img, x, y)
    map_img.paste(arrow_img, (int(x), int(y)), arrow_img)
    return map_img.convert("RGB")

def method_feature_extration(x, y, car_angle, car_size):
    state_img = getstate_image(x, y, car_angle, car_size)
    model = models.resnet18(pretrained=True)
    #model_1 = model.fc
    layers = list(model.children())[:-1]
    #    = nn.Sequential(*(model.layer1))
    model_1 = nn.Sequential(*layers)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data = data_transforms(state_img).float()
    data = data.unsqueeze_(0)
    features = model_1(data)
    return np.asarray(features.data)


