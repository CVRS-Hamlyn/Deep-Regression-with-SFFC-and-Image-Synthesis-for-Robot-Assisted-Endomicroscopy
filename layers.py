import torch 
import torch.nn as nn
import torch.nn.functional as F

class OrdinalRegressionLayer(nn.Module):
    def __init__(self, num_class):
        super(OrdinalRegressionLayer, self).__init__()
        self.num_class = num_class
        temp = torch.zeros((1,1))
        temp = F.pad(temp, (num_class, 0), 'constant', -1)
        self.mask = F.pad(temp, (0, num_class), 'constant', 1)

    def forward(self, x):
        """
        :Input x: NxC, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1
                 ord prob is the probability of each label, N x OrdNum 
        """
        batch_size, C = x.size()
        ord_num = C // 2
        x = x.view(-1, 2, ord_num)
        prob = F.log_softmax(x, dim=1)
        ord_prob = F.softmax(x, dim=1)[:, 0, :]
        ord_label = torch.sum((ord_prob > 0.5) * self.mask.to(x.device), dim=1)
        return prob, ord_prob, ord_label
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def exist_frame_select(inputs, frame_indexes):
    """
    :Input inputs: dictionary contains video: Nx201xHxW
    :Input frame_indexes: Nx2
    :return exist_frames: Nx2xHxW
    """
    video = inputs['video']
    bs, c, h, w = video.shape
    index_range = (torch.tensor(range(c)) + 1).to(device=video.device)
    # get index of selected frames
    index_1st = frame_indexes[:,0].unsqueeze(-1)
    index_2nd = frame_indexes[:,1].unsqueeze(-1)
    # one hot encode Nx201 frames
    onehot_1st = torch.where(index_range==index_1st+1, (index_1st+1) / index_range, index_1st * 0)
    onehot_2nd = torch.where(index_range==index_2nd+1, (index_2nd+1) / index_range, index_2nd * 0)
    # select frames by one hot encode * frames
    frame_1st = (video * onehot_1st.view(bs, -1, 1, 1)).sum(1).unsqueeze(1)
    frame_2nd = (video * onehot_1st.view(bs, -1, 1, 1)).sum(1).unsqueeze(1)
    # concat two frames to be interpolated
    exist_frames = torch.cat((frame_1st, frame_2nd), dim=1)
    return exist_frames


def floor_div(input, K):
    residual = torch.remainder(input, K)
    out = (input - residual) / K
    return out


def find_nearby_frame_indexes(inputs, distance, step_size):
    """
    :Input inputs: dictionary contains optimal: N (index of optimal frame), 
                                       index: N (index of input frames)
    : return: frame_indexes: Nx2
            : direction: Nx1 -1 or 1
            : ratio: Nx1
    """
    input_index = inputs['index'].unsqueeze(-1)

    direction = torch.sign(distance)

    num_indexes = floor_div(torch.abs(distance), step_size)
    residual = torch.abs(distance) % step_size

    ratio = residual / step_size

    index_pre = torch.maximum(torch.minimum((input_index + direction * num_indexes), inputs['num_frames'].unsqueeze(-1) - 1), torch.tensor([0]).to(device=input_index.device))
    index_post = torch.maximum(torch.minimum(input_index + direction * (num_indexes + 1), inputs['num_frames'].unsqueeze(-1) -1), torch.tensor([0]).to(device=input_index.device))

    frame_indexes = torch.cat((index_pre, index_post), dim=1)

    return frame_indexes, direction, ratio



class pCLE_interpolation(nn.Module):
    def __init__(self, batch_size, height, width, step_size):
        super(pCLE_interpolation, self).__init__()
        self.height = height
        self.width = width
        self.step_size = step_size

    def forward(self, exist_frames, ratio, direction):
        """
        :Input exist_frames: Nx2xHxW
        :Input ratio: Nx1 a scale value
        :Input direction: Nx1 -1 or 1
        :return: interp_frames: Nx1xHxW
        """
        bs, _, _, _ = exist_frames.shape
        interp_frames = torch.zeros(bs, 1, self.height, self.width).to(device=exist_frames.device)

        cond_1 = direction > 0
        cond_2 = direction < 0
        cond_1_frame = cond_1.unsqueeze(-1).unsqueeze(-1).repeat(1, 2, self.height, self.width)
        cond_2_frame = cond_2.unsqueeze(-1).unsqueeze(-1).repeat(1, 2, self.height, self.width)
        
        interp_frames[cond_1] = ratio[cond_1].view(-1, 1, 1) * exist_frames[cond_1_frame].view(-1, 2, self.height, self.width)[:, 1, :, :] + \
                               (1 - ratio[cond_1].view(-1, 1, 1)) * exist_frames[cond_1_frame].view(-1, 2, self.height, self.width)[:, 0, :, :]
        interp_frames[cond_2] = ratio[cond_2].view(-1, 1, 1) * exist_frames[cond_2_frame].view(-1, 2, self.height, self.width)[:, 0, :, :] + \
                               (1 - ratio[cond_2].view(-1, 1, 1)) * exist_frames[cond_2_frame].view(-1, 2, self.height, self.width)[:, 1, :, :]

        return interp_frames



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(21, 1)
        self.mu_y_pool   = nn.AvgPool2d(21, 1)
        self.sig_x_pool  = nn.AvgPool2d(21, 1)
        self.sig_y_pool  = nn.AvgPool2d(21, 1)
        self.sig_xy_pool = nn.AvgPool2d(21, 1)

        self.refl = nn.ReflectionPad2d(10)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class BM(nn.Module):
    def __init__(self):
        super(BM,self).__init__()
        self.aver_h = nn.AvgPool2d((1,9), 1)
        self.aver_v = nn.AvgPool2d((9,1), 1)
        self.pad_h = nn.ZeroPad2d((4, 4, 0, 0))
        self.pad_v = nn.ZeroPad2d((0, 0, 4, 4))
    
    def forward(self, x):
        x_h = self.pad_h(x)
        x_v = self.pad_v(x)

        B_hor = self.aver_h(x_h)
        B_ver = self.aver_v(x_v)

        D_F_ver = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        D_F_hor = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])

        D_B_ver = torch.abs(B_ver[:, :, :-1, :] - B_ver[:, :, 1:, :])
        D_B_hor = torch.abs(B_hor[:, :, :, :-1] - B_hor[:, :, :, 1:])

        T_ver = D_F_ver - D_B_ver
        T_hor = D_F_hor - D_B_hor

        V_ver = torch.maximum(T_ver, torch.tensor([0]).to(x.device))
        V_hor = torch.maximum(T_hor, torch.tensor([0]).to(x.device))

        S_V_ver = torch.sum(V_ver[:, :, 1:-1, 1:-1], dim=(-2, -1))
        S_V_hor = torch.sum(V_hor[:, :, 1:-1, 1:-1], dim=(-2, -1))
        
        blur = torch.maximum(S_V_ver, S_V_hor)

        return blur

        

