from functools import total_ordering
from random import gauss
import torch
import torch.utils.data as data
import os
import pickle
import json
import numpy as np
from torch.utils.data import DataLoader
import math
import random
import torch.nn as nn

def load_json(path):
    f = open(path, )
    dataset = json.load(f)
    return dataset

def read_pkl(path):
    with open(path, 'rb') as f:
        array = pickle.load(f)
    return array

class pcle_dataset(data.Dataset):
    def __init__(self,
                root_path,
                mode,
                num_channels,
                use_interp,
		        norm=True):
        super(pcle_dataset, self).__init__()
        self.root_path = root_path
        self.mode = mode
        if mode[:5] == "train":
            self.train = True
        else:
            self.train = False
        self.num_channels = num_channels
        self.use_interp = use_interp
        self.norm = norm
        json_path = ('./{}_dataset.json').format(mode)
        self.data_list = load_json(json_path)


    def normlization(self, data, I_max, I_min):
        if I_max == None:
            I_max = torch.max(data)
            I_min = torch.min(data)

        data_norm = (data - I_min) / (I_max - I_min)

        return data_norm

        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        inputs = {}
        data_dict = self.data_list[str(idx)]
        frame_split = data_dict['frame'].split('/')
        frame_path = os.path.join(self.root_path, frame_split[-3], frame_split[-2], frame_split[-1])
        video_path = os.path.join(self.root_path, frame_split[-3], frame_split[-2], 'video.npy')
        BM_path = os.path.join(self.root_path, frame_split[-3], frame_split[-2], 'BM.npy')
        video = torch.from_numpy(np.load(video_path)).float()
        BM_score = torch.from_numpy(np.load(BM_path)).float()
        num_frames, height, width = video.shape
        temp = torch.zeros(201, height, width)
        temp_BM = torch.zeros(161)
        temp[:num_frames, :, :] = video
        temp_BM[:num_frames] = BM_score
        inputs['video'] = temp
        inputs['BM'] = temp_BM
        inputs['index'] = torch.tensor(data_dict['index'], dtype=torch.long)
        inputs['num_frames'] = num_frames
        
        if self.use_interp == True:
            inputs['optimal'] = torch.tensor(data_dict['optimal_index'], dtype=torch.long)
            
            inputs['V_max'] = torch.max(video)
            inputs['V_min'] = -400
            inputs['num_frames'] = num_frames

            im_pcle = torch.from_numpy(read_pkl(frame_path)).type(torch.float32).unsqueeze(0)
            distance = data_dict['distance']
            inputs['distance'] = torch.tensor(distance, dtype=torch.float32)
        else:
            
            inputs['index'] = torch.tensor(data_dict['index'], dtype=torch.long)
            im_pcle = torch.from_numpy(read_pkl(frame_path)).unsqueeze(0).float()
            # im_pcle = self.conv(self.normlization(im_pcle, I_max=8191, I_min=-400)).squeeze(0)
            distance = data_dict['distance']
            inputs['distance'] = torch.tensor(distance)
        

        if self.norm == True:
            if self.num_channels == 2:
                ipt_ch1 = self.normlization(im_pcle, None, None)
                ipt_ch2 = self.normlization(im_pcle, I_max=8191, I_min=-400)
                inputs['frame'] = torch.cat((ipt_ch1, ipt_ch2), dim=0)
            elif self.num_channels == 3:
                inputs['frame'] = self.normlization(im_pcle, None, None).repeat(3, 1, 1)
        else:
            inputs['frame'] = im_pcle
        return inputs