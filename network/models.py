from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.serialization import load
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath


    

def mobilenet_v2(in_channels, out_channels, pretrained=False):
    model = models.mobilenet_v2()
    model.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)
    model.classifier[1] = nn.Linear(1280, out_channels, bias=True)
    
    model.apply(init_weights)


    return model


def alexnet(in_channels, out_channels, pretrained=False):
    model = models.alexnet()
    model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2))
    model.classifier[6] = nn.Linear(4096, out_channels, bias=True)

    model.apply(init_weights)


    return model

def vgg16(in_channels, out_channels, pretrained=False):
    model = models.vgg16()
    model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    model.classifier[6] = nn.Linear(4096, out_channels, bias=True)

    model.apply(init_weights)

    
    return model

def densenet(in_channels, out_channels, pretrained=False):
    model = models.densenet161()
    model.features.conv0 = nn.Conv2d(in_channels, 96, kernel_size=(7,7), stride=(2,2), padding=(3,3))
    model.classifier = nn.Linear(2208, out_channels, bias=True)

    model.apply(init_weights)

    
    return model

def resnet50(in_channels, out_channels, pretrained=False):
    model = models.resnet50()    
    model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
 
    # classifier = []
    # classifier.append(nn.Linear(512, 256, bias=True))
    # classifier.append(nn.Linear(256, out_channels, bias=True))
    # model.fc = nn.Sequential(*classifier)
 
    model.fc = nn.Linear(2048, out_channels, bias=True)

    model.apply(init_weights)



    return model

def resnext(in_channels, out_channels, pretrained=False):
    model = models.resnext50_32x4d()
    model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
 
    # classifier = []
    # classifier.append(nn.Linear(512, 256, bias=True))
    # classifier.append(nn.Linear(256, out_channels, bias=True))
    # model.fc = nn.Sequential(*classifier)
 
    model.fc = nn.Linear(2048, out_channels, bias=True)

    model.apply(init_weights)



    return model

def wide_resnet(in_channels, out_channels, pretrained=False):
    model = models.wide_resnet50_2()
    model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
 
    # classifier = []
    # classifier.append(nn.Linear(512, 256, bias=True))
    # classifier.append(nn.Linear(256, out_channels, bias=True))
    # model.fc = nn.Sequential(*classifier)
 
    model.fc = nn.Linear(2048, out_channels, bias=True)

    model.apply(init_weights)



    return model



class zhang(nn.Module):
    def __init__(self, in_channels, out_channels, pretrained=False):
        super(zhang, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrained = pretrained
        self.model = self.zhang_net(self.in_channels, self.out_channels, self.pretrained)

        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook

        self.model.classifier[0].register_forward_hook(get_activation('feature'))

        

    @staticmethod 
    def zhang_net(in_channels, out_channels, pretrained=False):
        model = models.mobilenet_v2()
        model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = nn.Linear(1280, out_channels, bias=True)
        # model.apply(init_weights)

        return model
    
    def forward(self, x):
        return self.model(x)
    
    def get_act(self):
        return self.activation


def Z_net(in_channels, out_channels, pretrained=False):
    model = zhang(in_channels, out_channels, pretrained)

    return model


# def Z_net(in_channels, out_channels, pretrained=False):
#     model = models.mobilenet_v2()
#     model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     model.classifier[1] = nn.Linear(1280, out_channels, bias=True)

#     return model


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

class Fourier_conv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(Fourier_conv, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        bs, c, h ,w = x.size()
        ffted = fft.rfftn(x, s=(h,w), dim=(-2, -1), norm='ortho')
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        
        ffted = torch.split(ffted, int(ffted.shape[1] / 2), dim=1)
        ffted = torch.complex(ffted[0],ffted[1])
        out = torch.fft.irfftn(ffted,s=(h,w),dim=(2,3),norm='ortho')
        
        return out

def resnet18(in_channels, out_channels, pretrained=False):
    model = models.resnet18()    
    model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
 
    # classifier = []
    # classifier.append(nn.Linear(512, 256, bias=True))
    # classifier.append(nn.Linear(256, out_channels, bias=True))
    # model.fc = nn.Sequential(*classifier)
 
    model.fc = nn.Linear(512, out_channels, bias=True)

    # model.apply(init_weights)

    
    if in_channels == 3 and pretrained:
        model_dict = model.state_dict()
        loaded = model_zoo.load_url("https://download.pytorch.org/models/resnet18-f37072fd.pth")
        loaded = {k: v for k, v in loaded.items() if k in model_dict and k[:2] != 'fc'}
        model_dict.update(loaded)
        model.load_state_dict(model_dict)

    return model




class FFT_ResNet18(nn.Module):
    def __init__(self, FFT_block, in_channels=3, out_channels=1, pretrained=False):
        super(FFT_ResNet18, self).__init__()
        
        self.num_ch_model = np.array([64, 64, 128, 256, 512])
        self.model = models.resnet18()
        self.pad = nn.ReflectionPad2d((0,0,7,7))
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
                                nn.Dropout(p=0.2),
                                nn.Linear(512 * 2, 256, bias=True),
                                nn.Linear(256, out_channels, bias=True)
                                )
        self.conv_1x1_rf0 = nn.Conv2d(self.num_ch_model[0], self.num_ch_model[0], kernel_size=1, bias=False)
        self.fft_conv0 = FFT_block(self.num_ch_model[0], self.num_ch_model[0])
        self.conv_1x1_fr0 = nn.Conv2d(self.num_ch_model[0] * 2, self.num_ch_model[0], kernel_size=1, bias=False)
        
        self.conv_1x1_rf1 = nn.Conv2d(self.num_ch_model[0] + self.num_ch_model[1], self.num_ch_model[1], kernel_size=1, bias=False)
        self.fft_conv1 = FFT_block(self.num_ch_model[1], self.num_ch_model[1])
        self.conv_1x1_fr1 = nn.Conv2d(self.num_ch_model[1] + self.num_ch_model[1], self.num_ch_model[1], kernel_size=1, bias=False)
        
        self.conv_1x1_rf2 = nn.Conv2d(self.num_ch_model[1] + self.num_ch_model[2], self.num_ch_model[2], kernel_size=1, bias=False)
        self.fft_conv2 = FFT_block(self.num_ch_model[2], self.num_ch_model[2])
        self.conv_1x1_fr2 = nn.Conv2d(self.num_ch_model[2] + self.num_ch_model[2], self.num_ch_model[2], kernel_size=1, bias=False)
        
        self.conv_1x1_rf3 = nn.Conv2d(self.num_ch_model[2] + self.num_ch_model[3], self.num_ch_model[3], kernel_size=1, bias=False)
        self.fft_conv3 = FFT_block(self.num_ch_model[3], self.num_ch_model[3])
        self.conv_1x1_fr3 = nn.Conv2d(self.num_ch_model[3] + self.num_ch_model[3], self.num_ch_model[3], kernel_size=1, bias=False)
        
        self.conv_1x1_rf4 = nn.Conv2d(self.num_ch_model[3] + self.num_ch_model[4], self.num_ch_model[4], kernel_size=1, bias=False)
        self.fft_conv4 = FFT_block(self.num_ch_model[4], self.num_ch_model[4])
        self.conv_1x1_fr4 = nn.Conv2d(self.num_ch_model[4] + self.num_ch_model[4], self.num_ch_model[4], kernel_size=1, bias=False)
        
        self.down_sample = nn.MaxPool2d((2,2), 2)
        self._reset_parameters()

    def _reset_parameters(self):
        
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x): 
        x = self.pad(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        
        fmap_0 = self.model.relu(x)
        fmap_rf_0 = self.conv_1x1_rf0(fmap_0)
        fmap_f_0 = self.fft_conv0(fmap_rf_0)
        fmap_fr_0 = torch.cat((fmap_0, fmap_f_0), dim=1)
        fmap_r_0 = self.conv_1x1_fr0(fmap_fr_0)
        fmap_f_ds_0 = self.down_sample(fmap_f_0)
        
        fmap_1 = self.model.layer1(self.model.maxpool(fmap_r_0))
        fmap_rf_1 = self.conv_1x1_rf1(torch.cat((fmap_f_ds_0, fmap_1), dim=1))
        fmap_f_1 = self.fft_conv1(fmap_rf_1)
        fmap_fr_1 = torch.cat((fmap_1, fmap_f_1), dim=1)
        fmap_r_1 = self.conv_1x1_fr1(fmap_fr_1)
        fmap_f_ds_1 = self.down_sample(fmap_f_1)
        
        fmap_2 = self.model.layer2(fmap_r_1)
        fmap_rf_2 = self.conv_1x1_rf2(torch.cat((fmap_f_ds_1, fmap_2), dim=1))
        fmap_f_2 = self.fft_conv2(fmap_rf_2)
        fmap_fr_2 = torch.cat((fmap_2, fmap_f_2), dim=1)
        fmap_r_2 = self.conv_1x1_fr2(fmap_fr_2)
        fmap_f_ds_2 = self.down_sample(fmap_f_2)
        
        fmap_3 = self.model.layer3(fmap_r_2)
        fmap_rf_3 = self.conv_1x1_rf3(torch.cat((fmap_f_ds_2, fmap_3), dim=1))
        fmap_f_3 = self.fft_conv3(fmap_rf_3)
        fmap_fr_3 = torch.cat((fmap_3, fmap_f_3), dim=1)
        fmap_r_3 = self.conv_1x1_fr3(fmap_fr_3)
        fmap_f_ds_3 = self.down_sample(fmap_f_3)
        
        fmap_4 = self.model.layer4(fmap_r_3)
        fmap_rf_4 = self.conv_1x1_rf4(torch.cat((fmap_f_ds_3, fmap_4), dim=1))
        fmap_f_4 = self.fft_conv4(fmap_rf_4)
        fmap_fr_4 = torch.cat((fmap_4, fmap_f_4), dim=1)
        fmap_r_4 = self.conv_1x1_fr4(fmap_fr_4)
        
        fvec_r = self.model.avgpool(fmap_r_4)
        fvec_f = self.model.avgpool(fmap_f_4)
        
        fvec = torch.cat((fvec_r, fvec_f), dim=1)
        fvec = torch.flatten(fvec, 1)
        
        out = self.model.fc(fvec)
        
        return out

def fft_resnet18(in_channels, out_channels, pretrained=False):
    model = FFT_ResNet18(Fourier_conv, in_channels, out_channels, pretrained)
    # model.apply(init_weights)

    return model

