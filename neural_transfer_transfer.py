 # -*- coding: utf-8 -*-
"""
File: neural_transfer.py
Created on Tue Apr 21 2020

@author: Ankit Bansal

=========================================================================
Implements classical Neural style transfer
=========================================================================
"""

import time
import torch
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt

from vgg import Vgg16
import utils


def get_image_features(vgg_forward, feature_map_layers, gram=False):
    # vgg_forward = model(utils.normalize_batch(input_img))
    if gram:
        feature_maps = [utils.gram_matrix(vgg_forward[feature]) for feature in feature_map_layers]
    else:
        feature_maps = [vgg_forward[feature] for feature in feature_map_layers]
    
    return feature_maps

def get_loss(input_feature_maps, target_feature_maps, loss_fn=torch.nn.MSELoss()):
    loss=0
    for input_feature_map, target_feature_map in zip(input_feature_maps, target_feature_maps):
        loss += loss_fn(input_feature_map, target_feature_map)
    
    return loss


device = 'gpu' if torch.cuda.is_available() else 'cpu'
img_size = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

content_loss_layers = ['relu4_2']
style_loss_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
vgg = Vgg16(content_layers=content_loss_layers, style_layers=style_loss_layers).eval()

loss_fn = torch.nn.MSELoss()

img_transformer = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor(),
])

content_image = utils.load_image(Path(r"images\content\chicago.jpg"))
style_image = utils.load_image(Path(r"images\styles\mosaic.jpg"))

# utils.imshow(content_image, title='Content Image')
# utils.imshow(style_image, title='Style Image')

content_image = img_transformer(content_image).unsqueeze(0)
style_image = img_transformer(style_image).unsqueeze(0)


content_feature_maps = get_image_features(vgg(utils.normalize_batch(content_image)), content_loss_layers)
style_feature_maps_gram = get_image_features(vgg(utils.normalize_batch(style_image)), style_loss_layers, gram=True)


input_img = content_image.clone()
utils.imshow(input_img, title='Input Image')

optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

epochs = 125
content_weight, style_weight = 1, 1000000

tic = time.time()
epoch =[0]
while epoch[0] < epochs:
    def closure():
        global tic
        optimizer.zero_grad()
        input_img.data.clamp_(min=0, max=1)

        input_img_vgg_forward = vgg(utils.normalize_batch(input_img))
        target_content_maps = get_image_features(input_img_vgg_forward, content_loss_layers)
        target_style_maps_gram = get_image_features(input_img_vgg_forward, style_loss_layers, gram=True)

        content_loss = content_weight*get_loss(content_feature_maps, target_content_maps, loss_fn=loss_fn)
        style_loss = style_weight*get_loss(style_feature_maps_gram, target_style_maps_gram, loss_fn=loss_fn)
        total_loss = content_loss + style_loss

        total_loss.backward(retain_graph=True)

        epoch[0] += 1
        if (epoch[0])%25 == 0:
            print("run {}:".format(epoch[0]))
            print('Style Loss : {:4f}, Content Loss: {:4f}, Total Loss: {:4f}'.format(
                        style_loss.item(), content_loss.item(), total_loss.item()))
            utils.imshow(input_img, title='Styled Image after {} runs'.format(epoch[0]))
            print("Time Taken: {:4f}".format(time.time()- tic))
            tic = time.time()
            print()
        return total_loss

    optimizer.step(closure)

input_img.data.clamp_(min=0, max=1)
utils.imshow(input_img, title='Styled Image after {} runs'.format(epoch[0]+1))
plt.show()

