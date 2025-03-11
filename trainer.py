from __future__ import absolute_import, division, print_function
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import data_loader
import network
from layers import *
from tqdm.autonotebook import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as D
import time
import json


class Trainer:
    def __init__(self, options):
        self.opts = options

        #   self.log_path = os.path.join(self.opts.log_directory, self.opts.model_type + '_' + self.opts.loss_regre)

        #   if not os.path.exists(self.log_path):
            #   os.makedirs(self.log_path)

        #   self.writer = SummaryWriter(self.log_path)
        self.path_model = os.path.join(self.opts.model_folder, self.opts.checkpoint_dir)
        self.error_path = os.path.join(self.path_model, 'error')
        self.pred_path = os.path.join(self.path_model, 'pred')
        self.pos_path = os.path.join(self.path_model, 'pos')
        self.BM_path = os.path.join(self.path_model, 'BM')
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)
        if not os.path.exists(self.error_path):
            os.makedirs(self.error_path)
        if not os.path.exists(self.pred_path):
            os.makedirs(self.pred_path)
        if not os.path.exists(self.pos_path):
            os.makedirs(self.pos_path)
        if not os.path.exists(self.BM_path):
            os.makedirs(self.BM_path)
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)

	    # self.network_list = {"resnet18": network.resnet18, "resnet34": network.resnet34, "resnet50": network.resnet50, "resnet101": network.resnet101}
        self.network_list = {"resnet18": network.resnet18, "mobilenet_v2": network.mobilenet_v2, 
                            "fft_resnet18": network.fft_resnet18}
        self.models = {}
        self.parameters_to_train = []
        self.results = {}
        self.dist_error = {}
        self.distance_pre = {}
        self.converg_pos = {}
        self.BM = {}
        for i in range(-400, 401, 5):
            self.dist_error[i] = []
            self.distance_pre[i] = []
            self.converg_pos[i] = {}
            self.BM[i] = {}
        self.writer = SummaryWriter(self.path_model)


        if not self.opts.no_cuda:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        
        self.out_channels = self.opts.out_channels
        self.discrete_value = None


        self.models["spatial"] = self.network_list[self.opts.model_type](self.opts.in_channels, self.out_channels, pretrained=self.opts.pretrained)
        if self.opts.multi_gpu == True:
            self.models["spatial"] = nn.DataParallel(self.models["spatial"])
        self.models["spatial"].to(self.device)
        self.parameters_to_train += list(self.models["spatial"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opts.learning_rate, weight_decay=self.opts.weight_decay)

        self.L1_loss = nn.L1Loss()
        self.KL_loss = nn.KLDivLoss(reduction='batchmean')
        num_parameters = count_parameters(self.models["spatial"])
        print("Training model named:\n  ", self.opts.model_type)
        print("Number of Parameters in regression model: {:.1f}M".format( num_parameters/ 1e6))
        print("Training is using:\n  ", self.device)
        if torch.cuda.is_available():
            print('Using GPU: {}'.format(torch.cuda.get_device_name()))

        print("Checkpoint address:", self.path_model)

        train_dataset = data_loader.pcle_dataset(self.opts.root_path, 'train', self.opts.in_channels, 
                                                 self.opts.use_interp, self.opts.norm)

        self.train_loader = DataLoader(
            train_dataset, self.opts.batch_size, True,
            num_workers=self.opts.num_workers, pin_memory=True, drop_last=False)


        val_dataset = data_loader.pcle_dataset(self.opts.root_path, 'test', self.opts.in_channels, 
                                                 self.opts.use_interp, self.opts.norm)
        self.val_loader = DataLoader(
            val_dataset, 1, False,
            num_workers=self.opts.num_workers, pin_memory=True, drop_last=False)
        
        if self.opts.cyclic_lr:
            self.model_lr_scheduler = optim.lr_scheduler.CyclicLR(self.model_optimizer, base_lr=1e-5, max_lr=1e-4, 
                                                                step_size_up=2*((len(train_dataset) // self.opts.batch_size) + 1),
                                                                cycle_momentum=False)
        else:
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opts.scheduler_step_size, 0.1)


        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        print("Using interpolation:", self.opts.use_interp)
        print("Distance Inference: {} * tanh(x)".format(self.opts.dis_range))
        # print("Using Gaussian Prob Inference:", self.opts.use_gaussian_infer)
        # print("Using MixUp data argumentation:", self.opts.mixup)
        print("Using pretrained:", self.opts.pretrained)
        print("Number of input channels:", self.opts.in_channels)
        print("Number of output channels:", self.out_channels)
        print("Interp loss: {:.1f} * MoI + {:.1f} * SSIM + {:.1f} * BM".format(self.opts.MoI_weight, self.opts.SSIM_weight, self.opts.BM_weight))
        # print("Loss for classification:", self.opts.loss_class)
        # print("Loss for regression:", self.opts.loss_regre)
        if self.opts.use_interp:
            self.interpolation = pCLE_interpolation(self.opts.batch_size, self.opts.height, self.opts.width, self.opts.step_size).to(self.device)
            self.ssim = SSIM().to(self.device)
            self.bm = BM().to(self.device)


    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        for self.epoch in range(self.opts.num_epochs):
            self.run_epoch()

    def variance_loss(self, distance):

        discrete_value = self.discrete_value.to(self.device)
        variance = ((distance / 100) - discrete_value) ** 2
        loss_var = torch.sum(variance * self.prob_dist, dim=-1)

        return loss_var.mean()

    def run_epoch(self):
        #   self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        Train_Loader = tqdm(self.train_loader)
        loss_total = []
        Acc = []
        MAE = []

        for batch_idx, inputs in enumerate(Train_Loader):
            self.batch_idx = batch_idx   
            losses, mae = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses.backward()
            self.model_optimizer.step()

            loss_total.append(losses.cpu().detach().numpy())
            MAE.append(mae.cpu().detach().numpy())
            Train_Loader.set_postfix(loss_train=np.mean(loss_total), MAE_train=np.mean(MAE), epoch=self.epoch)
            if self.opts.cyclic_lr == True:
                self.model_lr_scheduler.step()
            self.step += 1

        mean_loss = np.mean(loss_total)
        mean_MAE = np.mean(MAE)
        self.results['train_loss'] = mean_loss
        self.results['train_mae'] = mean_MAE
        self.writer.add_scalar('Loss/train', mean_loss, self.epoch)
        self.writer.add_scalar('Error/train', mean_MAE, self.epoch)

        self.test()

        self.save_checkpoint()

        if self.opts.cyclic_lr == False:
            self.model_lr_scheduler.step()



    def process_batch(self, inputs):

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        frame = inputs['frame']
        # print(frame.shape)
        outputs = self.models['spatial'](frame)

        
        distance = torch.tanh(outputs) * self.opts.dis_range
        #   sign = torch.where(sign == 0, torch.tensor(1).to(self.device), sign)
        if self.opts.use_interp == True:
            frame_indexes, direction, ratio = find_nearby_frame_indexes(inputs, distance, self.opts.step_size)
            exist_frames = exist_frame_select(inputs, frame_indexes)

            interp_frames = self.interpolation(exist_frames, ratio, direction)

            losses = self.compute_loss(inputs, interp_frames, distance)

            mae = self.L1_loss(distance.squeeze(-1), inputs['distance'])
                        
        else:
            losses = self.L1_loss(distance.squeeze(-1), inputs['distance'])
            mae = self.L1_loss(distance.squeeze(-1), inputs['distance'])
            # input_print(mae)
        # mae = torch.where(torch.abs(mae) <= 25, torch.tensor(0,dtype=torch.float64).to(self.device), mae - 25)
        return losses, mae





    def norm(self, data, I_max, I_min):
        if I_max == None:
            I_max = torch.max(data)
            I_min = torch.min(data)

        data_norm = (data - I_min) / (I_max - I_min)

        return data_norm


    def test(self):
        # self.load_checkpoint(self.opts.num_epochs)
        # self.epoch=self.opts.num_epochs
        Acc = []
        MAE = []
        Dis = []
        Steps = []
        self.dist_error = {}
        self.distance_pre = {}
        self.converg_pos = {}
        self.BM = {}
        for i in range(-400, 401, 5):
            self.dist_error[i] = []
            self.distance_pre[i] = []
            self.converg_pos[i] = {}
            self.BM[i] = {}

        self.set_eval()
        Test_Loader = tqdm(self.val_loader)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(Test_Loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                frame = inputs['frame']
                video = inputs['video']
                index_curr = inputs['index']
                BM = inputs['BM'][0]
                outputs = self.models['spatial'](frame)
                # Fvec_seq.append(outputs['fvec'])
                distance = torch.tanh(outputs) * self.opts.dis_range
                # if not self.opts.no_multi_step:
                #     pred_dist.append(distance.unsqueeze(0))
                dist_target = inputs['distance']
                count = 0
                id = len(self.converg_pos[int(dist_target.cpu().detach().numpy())])
                self.converg_pos[int(dist_target.cpu().detach().numpy())][id + 1] = []
                # track_folder = os.path.join(self.path_model, "img", "{}".format(dist_target.cpu().detach().numpy()[0]), "{}".format(id + 1))
                # if not os.path.exists(track_folder):
                #     os.makedirs(track_folder)
                # if self.opts.in_channels == 3:
                #     im = self.norm(frame[:,1,:,:], I_max=8191, I_min=-400)
                #     plt.imsave(os.path.join(track_folder, "0.png"), im[0,:,:].cpu().detach().numpy(), cmap='gray',vmin=0,vmax=1)
                # else:
                #     plt.imsave(os.path.join(track_folder, "0.png"), frame[0,1,:,:].cpu().detach().numpy(), cmap='gray',vmin=0,vmax=1)
                self.BM[int(dist_target.cpu().detach().numpy())][id + 1] = []
                # print(BM[index_curr.cpu().detach().numpy()[0]].cpu().detach().numpy())
                self.BM[int(dist_target.cpu().detach().numpy())][id + 1].append(float(BM[index_curr.cpu().detach().numpy()[0]].cpu().detach().numpy()))
                for step in range(1,11,1):
                    outputs = self.models['spatial'](frame)
                    if step == 1:
                        dist = torch.tanh(outputs) * self.opts.dis_range
                    else:
                        dist = torch.tanh(outputs) * self.opts.dis_range
                        
                    
                    dis_sign = torch.sign(dist)
                    step_incre = torch.div(torch.abs(dist), 5, rounding_mode='trunc').squeeze(-1)
                    residual_dis = (torch.abs(dist) % 5).squeeze(-1)
                    if step == 1:
                        loc_probe  = dist_target - dist
                    else:
                        loc_probe -= dist
                    
                    self.converg_pos[int(dist_target.cpu().detach().numpy())][id + 1].append(float(loc_probe.cpu().detach().numpy()[0][0]))

                    if residual_dis <= 2.5:
                        step_incre = step_incre
                    else:
                        step_incre += 1
                    if step == 1:
                        index_next = (index_curr + dis_sign * step_incre)
                    else:
                        index_next = (index_next + dis_sign * step_incre)
                    index_next = torch.maximum(torch.minimum(index_next, inputs['num_frames'].unsqueeze(-1) - 1), torch.tensor([0]).to(device=index_next.device))
                    index_next = int(index_next.cpu().detach().numpy()[0])
                    frame_next = video[:,index_next,:,:].unsqueeze(1)
                    self.BM[int(dist_target.cpu().detach().numpy())][id + 1].append(float(BM[index_next].cpu().detach().numpy()))

                    if self.opts.in_channels == 2:
                        ipt_ch1 = self.norm(frame_next, None, None)
                        ipt_ch2 = self.norm(frame_next, I_max=8191, I_min=-400)
                        frame = torch.cat((ipt_ch1, ipt_ch2), dim=1)
                    elif self.opts.in_channels == 3:
                        frame = self.norm(frame_next, None, None).repeat(1, 3, 1, 1)


                    # if self.opts.in_channels == 3:
                    #     im = self.norm(frame_next, I_max=8191, I_min=-400)
                    #     # input(im.shape)
                    #     plt.imsave(os.path.join(track_folder, "{}.png".format(step)), im[0,0,:,:].cpu().detach().numpy(), cmap='gray',vmin=0,vmax=1)
                    # else:
                    #     plt.imsave(os.path.join(track_folder, "{}.png".format(step)), frame[0,1,:,:].cpu().detach().numpy(), cmap='gray',vmin=0,vmax=1)
                    
                #   sign = torch.where(sign == 0, torch.tensor(1).to(self.device), sign)
                mae = self.L1_loss(distance.squeeze(-1), inputs['distance'])
                
                dist_target = inputs['distance'][0]
                self.dist_error[int(dist_target)].append(torch.abs(distance.squeeze(-1)[0] - dist_target).cpu().detach().numpy())
                self.distance_pre[int(dist_target)].append(distance.squeeze(-1)[0].cpu().detach().numpy())
                # mae = torch.where(torch.abs(mae) <= 25, torch.tensor(0,dtype=torch.float64).to(self.device), mae - 25)
                MAE.append(mae.cpu().detach().numpy())
                Dis.append((inputs['distance'] - distance.squeeze(-1)).cpu().detach().numpy())
                Test_Loader.set_postfix(MAE_test = np.mean(MAE),epoch=self.epoch)

            mean_MAE = np.mean(MAE)
            # print("The loaded checkpoint:", self.path_model)
            print("MAE:", mean_MAE)
            np.save(os.path.join(self.error_path, "{}.npy".format(self.epoch)), self.dist_error)
            np.save(os.path.join(self.pred_path, "{}.npy".format(self.epoch)), self.distance_pre)
            with open(os.path.join(self.pos_path, "{}.json".format(self.epoch)), 'w') as f_test:
                json.dump(self.converg_pos, f_test)
            with open(os.path.join(self.BM_path, "{}.json".format(self.epoch)), 'w') as f_BM_test:
                json.dump(self.BM, f_BM_test)
            
            df = pd.Series(self.dist_error).rename_axis('distance').reset_index(name='error')
            df = pd.merge(df, pd.Series(self.distance_pre).rename_axis('distance').reset_index(name='pred_dist'), on='distance')
            df['pred_mean'] = df.pred_dist.apply(lambda x: np.mean(np.array(x)))
            df['pred_median'] = df.pred_dist.apply(lambda x: np.median(np.array(x)))
            df['pred_std'] = df.pred_dist.apply(lambda x: np.std(np.array(x)))
            df['pred_n'] = df.pred_dist.apply(lambda x:(np.array(x).shape[0]))
            df['pred_standard_error'] = df['pred_std']/np.sqrt(df.pred_n)
            df['vector'] = 0-df['pred_mean']

            figure_error = self.plt_error_figure(df=df)
            self.writer.add_figure('figs/pred_to_GT', figure_error, self.epoch)
            dir_acc = self.compute_dir_acc(df=df)
            
            figure_converg, width, BM_SC, E_c = self.plt_convergence()
            self.writer.add_figure('figs/convergence', figure_converg, self.epoch)
            self.writer.add_scalar('Eval_Metrics/dir_acc', dir_acc, self.epoch)
            self.writer.add_scalar('Eval_Metrics/width', width, self.epoch)
            self.writer.add_scalar('Eval_Metrics/BM_SC', BM_SC, self.epoch)
            self.writer.add_scalar('Eval_Metrics/MAE_C', E_c, self.epoch)
            self.writer.add_scalar('Eval_Metrics/MAE_1st', mean_MAE, self.epoch)

            # self.results['test_mae'] = mean_MAE
            # 
        self.set_train()


    def plt_error_figure(self, df):
        fig = plt.figure()
        barlist = plt.bar(
            df.distance.values, df.vector.values, color='blue'
        )
        for bar in barlist:
            bar.set_color('b')
        plt.fill_between(df.distance.values, df.vector.values - df.pred_standard_error, df.vector.values + df.pred_standard_error,
                    color='gray', alpha=0.5
                        )
        plt.title("Predicted focusing distance against GT distance (Ours)", fontsize=24)
        plt.axvline(x=-35, ymin=0, color='green', linewidth=5)
        plt.axvline(x=35, ymin=0, color='green', linewidth=5, label='working range')
        plt.ylabel('Steps to move during Robotic Manipulation',fontsize=18)
        plt.xlabel('GT distance',fontsize=18)

        plt.xlim=(-400, 400)
        plt.ylim=(-400, 400)
        x1, y1 = [-400, 400], [+400, -400]
        plt.plot(x1, y1, marker = 'o', color='r',label='ground truth', linewidth=5)
        plt.legend(fontsize=18)

        return fig

    def compute_dir_acc(self, df):
        # count from above
        count = 0

        # negative side count
        for i in range(80):
            distance_value = df.distance.values[i]
            vector_value = df.pred_dist[i]
            for j in range(len(vector_value)):
                if -1 * vector_value[j]>=0:
                    count+=1
                
        # positive side count
        for i in range(80, len(df.distance.values), 1):
            distance_value = df.distance.values[i]
            vector_value = df.pred_dist[i]
            for j in range(len(vector_value)):
                if -1 * vector_value[j] <= 0:
                    count+=1
        return count / 1706
    
    def plt_convergence(self):
        B_up = []
        B_down = []
        Pos = []
        BM_score = []
        fig = plt.figure()
        plt.xlim=(0, 30)
        plt.ylim=(-400, 400)
        for n, key in enumerate(self.converg_pos.keys()):
        #     if (int(key) >= -35) and (int(key) <= 35):
            dis_in = np.zeros((len(self.converg_pos[key]), 11))
            BM_in = np.zeros((len(self.BM[key]), 11))
            for i in range(len(self.converg_pos[key])):
                temp = self.converg_pos[key][i+1]
                temp_BM = self.BM[key][i+1]
                for j in range(11):
                    if j == 0:
                        dis_in[i,j] = int(key)
                        BM_in[i,j] = temp_BM[j-1]
                    else:
                        dis_in[i,j] = temp[j-1]
                        BM_in[i,j] = temp_BM[j-1]
            Pos.append(dis_in)
            BM_score.append(BM_in)
            med = np.median(dis_in, axis=0)
            std_error = np.std(dis_in, axis=0) / np.sqrt(dis_in.shape[0])
            plt.plot(med)
            B_up.append(med +std_error)
            B_down.append(med - std_error)
            plt.fill_between(range(11), med - std_error, med +std_error, alpha=0.2)
        B_up = np.array(B_up)
        B_down = np.array(B_down)
        max_b_up = np.max(B_up[:, 5:])
        max_b_down = np.min(B_down[:, 5:])
        width = max_b_up - max_b_down

        loc = np.concatenate(Pos)
        score = np.concatenate(BM_score)
        later_pos = loc[:, 5:]
        later_score = score[:, 5:]
        BM_SC_mu = np.mean(later_score, axis=0)
        BM_SC_std = np.std(BM_SC_mu)
        print("Samples in Range: {:.3f} -/+ {:.3f}".format(np.mean(BM_SC_mu), BM_SC_std))

        mu_pos = np.mean(np.abs(later_pos),axis=0)
        print("Convergence Error: {:.2f} -/+ {:.2f}".format(np.mean(mu_pos), np.std(mu_pos)))
        # print([385]+pos['385']['1'])
        plt.axhline(y=-35, xmin=0, color='green', linewidth=5)
        plt.axhline(y=35, xmin=0, color='green', linewidth=5, label='working range')
        plt.axhline(y=max_b_up, xmin=0, color='red', linestyle='dotted', linewidth=5)
        plt.axhline(y=max_b_down, xmin=0, color='red', linestyle='dotted', linewidth=5, label='upper-lower bound')
        plt.fill_between(range(5,11, 1), np.ones((6))*max_b_down, np.ones((6))*max_b_up, color='red', alpha=0.4)
        plt.title("Convergence and Upper-Lower Bounds (Ours)", fontsize=24)
        plt.ylabel('The Position of pCLE',fontsize=18)
        plt.xlabel('Steps',fontsize=18)
        plt.legend(fontsize=18)
        
        return fig, width, np.mean(BM_SC_mu), np.mean(mu_pos)


    def normlization(self, data):
        I_max = torch.amax(data, dim=(1,2,3))
        I_min = torch.amin(data, dim=(1,2,3))

        data_norm = (data - I_min.view(-1, 1, 1, 1)) / (I_max.view(-1, 1, 1, 1) - I_min.view(-1, 1, 1, 1))

        return data_norm



    def compute_loss(self, inputs, interp, distance):

        distance_t = inputs["distance"]
        video = inputs['video']
        bs, c, h, w = video.shape
        video = video.view(-1, self.opts.height, self.opts.width)

        # input_idx = inputs['index']
        # index_in = torch.tensor(range(input_idx.shape[0])).to(self.device)
        # input_idx = torch.where(input_idx < 0, input_idx, input_idx + index_in * c)

        optimal_idx = inputs['optimal']
        index_optimal = torch.tensor(range(optimal_idx.shape[0])).to(self.device)
        optimal_idx = torch.where(optimal_idx < 0, optimal_idx, optimal_idx + index_optimal * c)

        # frames_in = torch.index_select(video, 0, input_idx).unsqueeze(1)
        frames_opt = torch.index_select(video, 0, optimal_idx).unsqueeze(1)

        loss_regre = self.opts.l1_weight * self.L1_loss(distance.squeeze(-1), distance_t)
        
        loss_ssim = self.ssim(self.normlization(interp), self.normlization(frames_opt))

        loss_MoI = self.L1_loss(((interp - inputs['V_min'].view(-1, 1, 1, 1)) / (inputs['V_max'].view(-1, 1, 1, 1) - inputs['V_min'].view(-1, 1, 1, 1))).mean(dim=(1,2,3)),
                                ((frames_opt - inputs['V_min'].view(-1, 1, 1, 1)) / (inputs['V_max'].view(-1, 1, 1, 1) - inputs['V_min'].view(-1, 1, 1, 1))).mean(dim=(1,2,3)))
        loss_interp = self.opts.MoI_weight * loss_MoI + self.opts.SSIM_weight * loss_ssim.mean()

        # losses = loss_interp + loss_regre
        losses = loss_regre + loss_interp
        
        return losses

    



    # def ordinal_regression_loss(self, inputs, outputs):
    #     prob_pred, prob_c0, label = self.ordinal_regression_layer(outputs)
    #     bs = label.size()
    #     distance = torch.empty(bs).to(self.device)
    #     prob_target = inputs["ord_prob"]
    #     losses = (-1 * prob_target * prob_pred).sum(dim=1)
        
    #     condition_1 = label == 0
    #     condition_2 = torch.abs(label) == self.opts.ord_num
    #     condition_3 = ~ torch.logical_or(condition_1, condition_2)
        
    #     distance[condition_1] = torch.exp(torch.log(self.beta) * label[condition_1] / self.opts.ord_num) - 1
    #     distance[condition_2] = torch.sign(label[condition_2]) * torch.exp(torch.log(self.beta) * torch.abs(label[condition_2]) / self.opts.ord_num)
    #     t0 = torch.sign(label[condition_3]) * torch.exp(torch.log(self.beta) * torch.abs(label[condition_3]) / self.opts.ord_num)
    #     t1 = torch.sign(label[condition_3]) * torch.exp(torch.log(self.beta) * torch.abs(label[condition_3] + torch.sign(label[condition_3])) / self.opts.ord_num)
    #     distance[condition_3] = (t0 + t1) / 2

    #     mae = torch.abs(inputs['distance'] - distance).mean()
    #     class_pred = (prob_c0 > 0.5) * 1
    #     class_target = prob_target[:, 0, :]
    #     acc = torch.where((class_pred - class_target) == 0, 
    #                     torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device)).mean()

    #     return losses.mean(), acc, mae



    # def categorical_crossentropy(self, prob_t, prob):
    #     loss = - (prob_t[:, 0] * (torch.log10(prob[:, 0]) + 1e-10) +
    #               prob_t[:, 1] * (torch.log10(prob[:, 1]) + 1e-10) +
    #               prob_t[:, 2] * (torch.log10(prob[:, 2]) + 1e-10)).mean()

    #     return loss


    def save_checkpoint(self):
        PATH = os.path.join(self.path_model, ('model_{}.pt').format(self.epoch))

        torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.models['spatial'].state_dict(),
                    'optimizer_state_dict': self.model_optimizer.state_dict(),
                    }, PATH)


    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.path_model, ('model_{}.pt').format(epoch)))
        # input(checkpoint['model_state_dict'].keys())
        for key in list(checkpoint['model_state_dict'].keys()):
            checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
        self.models['spatial'].load_state_dict(checkpoint['model_state_dict'])
        # self.model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # input(checkpoint['model_state_dict'].keys())
