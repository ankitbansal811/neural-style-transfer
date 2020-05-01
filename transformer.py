 # -*- coding: utf-8 -*-
"""
File: transformer.py
Created on Sun Apr 26 2020

@author: Ankit Bansal

=========================================================================
Image transformer as presented by Justin Johnson in Fast Neural Transfer Paper
=========================================================================
"""

import torch.nn as nn

# For model achitecture refer https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf

class Transformer(nn.Module):
    """
    Image transformer model has three stages 
    Downsamplig - Residual Blocks - Upsampling
    These models have the same out shape same as input shape.
    Authors: 'Our image transformation networks roughly follow the architectural guidelines
    set forth by Radford et al [42]. We do not use any pooling layers, instead using
    strided and fractionally strided convolutions for in-network downsampling and
    upsampling. Our network body consists of five residual blocks [43] using the architecture of [44]. All non-residual convolutional layers are followed by spatial
    batch normalization [45] and ReLU nonlinearities with the exception of the output layer, which instead uses a scaled tanh to ensure that the output image has
    pixels in the range [0, 255].'
    """
    def __init__(self, print_log=False):
        super(Transformer, self).__init__()
        self.print_log = print_log

        self.Downsampling = nn.Sequential()
        self.Downsampling.add_module(name='conv1', module=ConvLayer(3, 32, 9, 1))
        self.Downsampling.add_module('conv2', ConvLayer(32, 64, 3, 2))
        self.Downsampling.add_module('conv3', ConvLayer(64, 128, 3, 2))

        self.ResBlocks = nn.Sequential()
        self.ResBlocks.add_module('ResBlock1', ResidualBlock(128))
        self.ResBlocks.add_module('ResBlock2', ResidualBlock(128))
        self.ResBlocks.add_module('ResBlock3', ResidualBlock(128))
        self.ResBlocks.add_module('ResBlock4', ResidualBlock(128))
        self.ResBlocks.add_module('ResBlock5', ResidualBlock(128))

        self.Upsampling = nn.Sequential()
        self.Upsampling.add_module('Upsample1', UpsampleLayer(128, 64, 3, 1, upsample=2))
        self.Upsampling.add_module('Upsample2', UpsampleLayer(64, 32, 3, 1, upsample=2))

        # In the end we a simple Conv2D layer followed by tanh activation
        self.last_layer = nn.Conv2d(32, 3, 9, 1, 9//2)

    def forward(self, X):
        if self.print_log: print("Input batch shape:", X.shape)
        X = self.Downsampling(X)
        if self.print_log: print("Downsampled batch shape:", X.shape)
        X = self.ResBlocks(X)
        if self.print_log: print("ResBlocks batch shape:", X.shape)
        X = self.Upsampling(X)
        if self.print_log: print("Upsampled batch shape:", X.shape)
        X = self.last_layer(X)
        return X


class Norm(nn.Module):
    """ A common normalization class to be used. 
    This will facilitate easy change in (all) normalization layers."""
    def __init__(self, num_features, affine=True):
        super(Norm, self).__init__()
        # self.norm = nn.InstanceNorm2d(num_features, affine=affine)               # Uncomment to use instance normalization
        self.norm = nn.BatchNorm2d(num_features, affine=affine)                    # Uncomment to use instance normalization

    def forward(self, X):
        out = self.norm(X)
        return out


class ConvLayer(nn.Module):
    """
    Downsampling Conv Layer with Normalization and Relu Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
        self.model.add_module('norm', Norm(num_features=out_channels))
        self.model.add_module('relu', nn.ReLU())

    def forward(self, X):
        out = self.model(X)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block do not scale the image or channels rather try to find important features.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential()
        # We are going to use zero-padding instead of reflection padding
        self.model.add_module('conv1', nn.Conv2d(in_channels=channels, out_channels=channels, 
                                                        kernel_size=3, stride=1, padding=1))
        self.model.add_module('norm1', Norm(num_features=channels))
        self.model.add_module('relu', nn.ReLU())
        self.model.add_module('conv2', nn.Conv2d(channels, channels, 3, 1, padding=1))
        self.model.add_module('norm2', Norm(channels))

    def forward(self, X):
        residual = X
        out = self.model(X)
        out = out + residual
        return out


class UpsampleLayer(nn.Module):
    """
    These layers will increase image Height and Width to the original image and decrease number of channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample):
        super(UpsampleLayer, self).__init__()
        self.upsample = upsample
        # Upsamplig layers are same as downsampling, the only difference is of the input which is scaled in forward pass.
        self.model = ConvLayer(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, X):
        X = nn.Upsample(scale_factor=self.upsample, mode='nearest')(X)
        out = self.model(X)
        return out


if __name__ == "__main__":
    import torch
    img = torch.randn(1, 3, 256, 256)
    model = Transformer(print_log=True)
    print(model(img).shape)
