 # -*- coding: utf-8 -*-
"""
File: vgg.py
Created on Tue Apr 21 2020

@author: Ankit Bansal

=========================================================================
Loads pretrained VGG19 model and provides a class for style propogation
=========================================================================
"""

import torch
from torchvision import models

class Vgg16(torch.nn.Module):
    """
    nn.Module class to load a pretrained VGG16 network and propogate image through feature layers.
    Forward pass through this will return feature maps from different layers as a dict.

    Parameters
    ----------
    requires_grad : boolean[False]
        Weather or not to train the vgg model weights
    content_layers : list(str)
        List of layer names which will be used for content loss calculations
    style_layers : list(str)
        List of layer(s) names which will be used for style loss calculations
    avg_pooling : boolean[False]
        Whether to use Average pooling (if True) or the default Max pooling (if False)
    """
    def __init__(self, requires_grad=False, content_layers=['relu4_2'],
                 style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                 avg_pooling=False):
        
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        i = 1  # increment every time we see a pooling layer
        j = 0  # increment every time we see a conv and reset with every pooling layer
        model = torch.nn.Sequential()
        self.model_parts = []
        for layer in vgg_pretrained_features.children():
            if isinstance(layer, torch.nn.Conv2d):
                j += 1
                name = 'conv{}_{}'.format(i, j)
            elif isinstance(layer, torch.nn.ReLU):
                name = 'relu{}_{}'.format(i, j)
                # layer = nn.ReLU(inplace=False)
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                i += 1
                j = 0
                # Comment out below three rows to keep the default MaxPool2d layers
                if avg_pooling:
                    # Assuming that a pooling layer output will not be used in loss
                    layer = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
                    model.add_module(name, layer)
                    continue;
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = 'bn{}_{}'.format(i, j)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)
            
            if (name in content_layers + style_layers):
                # break the sequence if the current layer output is used as a output
                # initialize a new model, and append it to the list of models 
                # print(name)
                self.model_parts.append(model)
                model = torch.nn.Sequential()
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, X):
        out = {}
        for model in self.model_parts:
            X = model(X)
            out[list(model.named_children())[-1][0]] = X
        return out