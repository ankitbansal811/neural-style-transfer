 # -*- coding: utf-8 -*-
"""
File: utils.py
Created on Tue Apr 21 2020

@author: Ankit Bansal

=========================================================================
Utility function for neural style transfer
=========================================================================
"""

import torch
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

def load_image(filename, size=None, scale=None):
    """
    Loads a give filepath image and returns PIL image which can be used with torchvision transforms.
    Resize the input image based on size and/or scale.

    Parameters
    ----------
    filename : Path/ str
        Path to the input image
    size : int
        Expected output size (pixel) of the image
    scale : float/int
        Scaling factor for the image

    Reutrns
    ---------
    img : PIL image
    """
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.ANTIALIAS)
    return img


def save_tensor_image(filename, data):
    """
    given a image tensor (3 X H X W) saves it at filename path
    """
    img = transforms.ToPILImage()(data)
    img.save(filename)
    

def normalize_batch(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Given a batch of images normalize them using given mean and std.
    Default mean and std are that of VGG16 network.

    batch : torch.Tensor(b, C, H, W) - scaled between 0-1
    """
    # normalize using imagenet mean and std
    batch = batch.clone()
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    # if your image data is scaled to scale 0-255, uncomment the line below
    # batch.div_(255.0)
    return (batch - mean) / std


def gram_matrix(feature_map):
    """Converts a feature map to gram matrix."""
    (b, ch, h, w) = feature_map.size()
    features = feature_map.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def imshow(image, title=None):
    """Show an image in matplotlib plot. Input can be PIL image or Tensor"""
    if isinstance(image, torch.Tensor):    
        image = image.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = transforms.ToPILImage()(image)    # convert tensor to PIL image

    plt.figure()
    plt.imshow(image)             # show PIL image 
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    # cv2.imshow(title, image)

