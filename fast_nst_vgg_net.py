import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VggNet(nn.Module):
    def __init__(self):
        super(VggNet, self).__init__()
        # Load the pre-trained VGG16 model
        # self.vgg = models.vgg16(pretrained=True).features.to(device).eval()
        # self.vgg = models.vgg19(weights=
        #                         models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
        print(self.vgg)

    def forward(self, x):
        # Define the dictionary to store extracted features
        features = {}
        # iterate each layers in the VGG16 model
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in {'0', '5', '10', '19', '28'}:
                # extract the output from each ReLU activation function
                features[name] = x
        # return extracted features from specified layers
        return features