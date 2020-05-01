 # -*- coding: utf-8 -*-
"""
File: fast_NST_demo.py
Created on Fri May 01 2020

@author: Ankit Bansal

=========================================================================
Run inference (styling) on Fast NST model. 
=========================================================================
"""
import os
import sys
import argparse
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import utils
from transformer import Transformer

def style_image(args):
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    transformer = Transformer()
    state_dict = torch.load(args.model)
    transformer.load_state_dict(state_dict)
    transformer.to(device)

    content_img = utils.load_image(args.image, scale=args.scale)
    img_to_tensor = transforms.ToTensor()
    content_img = img_to_tensor(content_img).unsqueeze(0).to(device)               # convert to tensor and add fake batch dimentsion

    output = transformer(content_img)

    if args.show:
        utils.imshow(output)
        plt.waitforbuttonpress(0)
        plt.close()
    
    if args.output_path:
        utils.save_tensor_image(args.output_path, output[0])
    
    return True

def arg_parser():
    """
    Parse input parameters
    """
    parser = argparse.ArgumentParser(description="Fast NST demo (styler)")

    parser.add_argument('--model', type=str, required=True, help="Saved Fast NST (pytorch) model - Path to .pth file")
    parser.add_argument('--image', type=str, required=True, help="(Path of) Image which is to be styled")
    parser.add_argument('--output-path', type=str, required=False, default=None, help="path where styled image will be saved"
    " will not be saved if path is not passed instead use a pop-up window")
    parser.add_argument('--scale', type=float, required=False, default=None, help="Scaling factor for input image."
    "Use <1 for scaling down, >1 for scaling up")
    parser.add_argument('--show', action='store_true', help="pass this if you want to visualize styled image")

    return parser.parse_args()


def check_out(args):
    if not (args.show or args.output_path):
        print("ERROR: Either use --show flag or pass an output-path to save the styled image")
        sys.exit(1)


if __name__ == "__main__":
    args = arg_parser()
    check_out(args)
    style_image(args)
