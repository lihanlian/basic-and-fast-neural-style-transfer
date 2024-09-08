import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        # set amount of padding to be half of kernel size
        reflection_padding = kernel_size // 2
        # applied before conv2d to ensure the conolution does not reduce the spatial dimensions of the tensor
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # construct standard convolutional layer
        self.conv1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)
        # normalize the input tensor (each instance in the batch individually) to have zero mean and unit vairance
        self.in1 = torch.nn.InstanceNorm2d(num_features=channels, affine=True)

        self.conv2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(num_features=channels, affine=True)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        # add residual to the output tensor from two conv layers, 
        # which helps in mitigating the vanishing gradient problems
        out = out + residual
        return out
    
class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.relfection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            # increase the spatial dimensions of the feature maps (x_in) by the specified scale factor
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)

        out = self.relfection_pad(x_in)
        # after upsampling and padding, perform convolution operation
        out = self.conv2d(out)
        return out

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(in_channels=3, out_channels=32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(num_features=32, affine=True)
        # stride=2, reduce dimension by 1/2
        self.conv2 = ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(num_features=64, affine=True)
        # stride=2, reduce dimension by 1/2
        self.conv3 = ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(num_features=128, affine=True)

        # Residula layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling layers
        # upsample=2, increase dimension by 2
        self.deconv1 = UpsampleConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(num_features=64, affine=True)
        # upsample=2, increase dimension by 2
        self.deconv2 = UpsampleConvLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(num_features=32, affine=True)
        self.deconv3 = ConvLayer(in_channels=32, out_channels=3, kernel_size=9, stride=1)

        # Nonlinear activation layer
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Apply initial convolution layers
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))

        # Apply residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # Apply upsampling layers
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2((x))))
        x = self.deconv3(x)

        return x