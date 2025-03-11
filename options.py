from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(os.getcwd()) # the directory that options.py resides in

print(file_dir)
data_folder = file_dir
model_folder = os.path.join(file_dir, 'Chi_models', 'filtered_data')
class AutoFocusOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="AutoFocus options")

        # PATHS
        self.parser.add_argument("--root_path",
                                 type=str,
                                 help="The root path",
                                 default=data_folder)
        self.parser.add_argument("--model_folder",
                                 type=str,
                                 help="The folder for storing models",
                                 default=model_folder)
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="log_book")
        self.parser.add_argument("--checkpoint_dir",
                                 type=str,
                                 help="The checkpoint directory",
                                 default='SFFC-Net')
        self.parser.add_argument("--onehot_model_dir",
                                 type=str,
                                 help="trained one-hot encoder model directory",
                                 default='one_hot_encoder')
        # DATA FEATURES
        self.parser.add_argument("--width",
                                 type=int,
                                 help="width of pCLE image",
                                 default=384)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="height of pCLE image",
                                 default=274)
        self.parser.add_argument("--step_size",
                                 type=int,
                                 help="step size of data collection",
                                 default=5)
        # TRAINING options
        self.parser.add_argument("--in_channels",
				                 type=int,
				                 help="number of input channels",
				                 default=2)
        self.parser.add_argument("--out_channels",
                                 type=int,
                                 help="number of output channels",
                                 default=1)
        self.parser.add_argument("--num_classes",
                                 type=int,
                                 help="the number of classes for one-hot encoder",
                                 default=201)
        self.parser.add_argument("--dis_range",
                                 type=int,
                                 help="range of distance for regression",
                                 default=400)
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="The type of model",
                                 default="resnet18")
        self.parser.add_argument("--norm",
                                 help="whether apply normalization to video and image",
                                 action="store_true")
        self.parser.add_argument("--sigma_2",
                                 type=float,
                                 help="square variation of gaussian distribution",
                                 default=0.2)
        self.parser.add_argument("--num_disc",
                                 type=int,
                                 help="number of discrete values within the distance range",
                                 default=81)
        self.parser.add_argument("--use_interp",
                                 help="whether use frame interpolation",
                                 action="store_true")
    
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--one_hot_lr",
                                 type=float,
                                 help="learning rate for one-hot encoder",
                                 default=1)
        self.parser.add_argument("--cyclic_lr",
                                 help="if set use cyclical learning rate",
                                 action="store_true")
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="regularization",
                                 default=1e-2)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=24)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--l1_weight",
                                 type=float,
                                 help="the weight of MAE to balance the loss function",
                                 default=1e-1)
        self.parser.add_argument("--var_weight",
                                 type=float,
                                 help="the weight of variance loss to balance loss function",
                                 default=5e-2)
        self.parser.add_argument("--MoI_weight",
                                 type=float,
                                 help="The weight of MoI loss",
                                 default=1)
        self.parser.add_argument("--SSIM_weight",
                                 type=float,
                                 help="The weight of SSIM loss",
                                 default=1)
        self.parser.add_argument("--BM_weight",
                                 type=float,
                                 help="The weight of blurry metric loss",
                                 default=0)


        # ABLATION options
        self.parser.add_argument("--pretrained",
                                 help="if set use pretrained weights",
                                 action="store_true")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)
        self.parser.add_argument("--multi_gpu",
                                 help="if set use multi_gpu",
                                 action="store_true")
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


