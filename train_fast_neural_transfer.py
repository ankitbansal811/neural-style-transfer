 # -*- coding: utf-8 -*-
"""
File: train_fast_neural_transfer.py
Created on Fri May 01 2020

@author: Ankit Bansal

=========================================================================
Trains a fast neural style transfer model
=========================================================================
"""
import argparse
import os
import time

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


from transformer import Transformer
from vgg import Vgg16
import utils

def get_image_features(vgg_forward, feature_map_layers, gram=False):
    # vgg_forward = model(utils.normalize_batch(input_img))
    if gram:
        feature_maps = [utils.gram_matrix(vgg_forward[feature]) for feature in feature_map_layers]
    else:
        feature_maps = [vgg_forward[feature] for feature in feature_map_layers]
    
    return feature_maps

def train(args):
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    img_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(args.dataset, img_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    model = Transformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_fn = torch.nn.MSELoss()

    content_loss_layers = ['relu3_3']
    style_loss_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
    loss_network = Vgg16(requires_grad=False, content_layers=content_loss_layers, style_layers=style_loss_layers).eval()             # represented as phi

    style_img = utils.load_image(args.style_image)
    style_img = img_transform(style_img)                                # y_s in paper: Style Target
    style_img = style_img.repeat(args.batch_size, 1, 1, 1).to(device)

    style_img_features = loss_network(utils.normalize_batch(style_img))
    style_gram_features = get_image_features(style_img_features, style_loss_layers, gram=True)

    for epoch in range(args.epochs):
        model.train()
        agg_content_loss, agg_style_loss = 0, 0
        count = 0

        for batch_id, (y_c, _) in enumerate(train_loader):
            batch_len = len(y_c)
            count += batch_len
            optimizer.zero_grad()

            y_c = y_c.to(device)                                        # content target
            y_hat = model(y_c)                                          # styled image

            # pass both content target and styled image to loss network to compute the perpectual losses
            # loss network (VGG16) accepts normalized data so normalize as well
            y_c = loss_network(utils.normalize_batch(y_c))
            y_hat = loss_network(utils.normalize_batch(y_hat))

            # Output of Vgg network is a dictionary of multiple layers, select only the layers associated with content loss
            y_c = get_image_features(y_c, content_loss_layers, gram=False)
            y_hat_content = get_image_features(y_hat, content_loss_layers, gram=False)

            content_loss = 0
            for predicted_contnet, content_feature in zip(y_hat_content, y_c):
                content_loss += loss_fn(predicted_contnet, content_feature) * args.content_weight

            y_hat_style = get_image_features(y_hat, style_loss_layers, gram=True)               # style layer feature maps for styled image
            style_loss = 0
            for predicted_gram, style_gram in zip(y_hat_style, style_gram_features):
                style_loss += loss_fn(predicted_gram, style_gram[:batch_len, :, :])             # batch size can be less than args.batch_size
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (count) % args.log_interval == 0:
                log_msg = "Time: {},\t Epoch: {},\t Processed: [{}/{}], \t\
                    Content Loss: {:.3},\tStyle Loss: {:.3},\tTotal Loss: {:.3}".format(time.ctime(), 
                epoch, count, len(train_dataset), agg_content_loss/count,
                agg_style_loss/count, (agg_content_loss+agg_style_loss)/count)

                print(log_msg)

        if (epoch+1)%args.checkpoint_interval == 0:
            model.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(epoch + 1) + ".pth"
            ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
            torch.save(model.state_dict(), ckpt_model_path)
            model.to(device).train()

    # save model
    model.eval().cpu()
    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    save_model_filename = "epoch_" + str(args.epochs) + "_" + timestamp + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def arg_parser():
    """
    Parse input arguments and build help at the same time.
    """
    parser = argparse.ArgumentParser(description="parser for fast-neural-style training")

    parser.add_argument("--epochs", type=int, default=10,
                                  help="number of training epochs, default is 2")
    parser.add_argument("--batch-size", type=int, default=2,
                                  help="batch size for training, default is 4")
    parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    parser.add_argument("--style-image", type=str, default="images/styles/mosaic.jpg",
                                  help="path to style-image")
    parser.add_argument("--save-model-dir", type=str, default='models',
                                  help="path to folder where trained model will be saved.")
    parser.add_argument("--checkpoint-model-dir", type=str, default='models',
                                  help="path to folder where checkpoints of trained models will be saved")
    parser.add_argument("--image-size", type=int, default=128,
                                  help="size of training images, default is 256 X 256")
    # parser.add_argument("--style-size", type=int, default=None,
    #                               help="size of style-image, default is the original size of style image")
    parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    parser.add_argument("--content-weight", type=float, default=1,
                                  help="weight for content-loss, default is 1")
    parser.add_argument("--style-weight", type=float, default=1e5,
                                  help="weight for style-loss, default is 1e5")
    parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    parser.add_argument("--log-interval", type=int, default=2,
                                  help="number of images after which the training loss is logged, default is 500")
    parser.add_argument("--checkpoint-interval", type=int, default=1,
                                  help="number of epochs after which a checkpoint of the trained model will be created")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    train(args)
