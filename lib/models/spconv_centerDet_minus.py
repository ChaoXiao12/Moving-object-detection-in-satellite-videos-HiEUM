import torch
import torch.nn as nn
from functools import partial
import numpy as np

from lib.models.spconv_unet import UNetV2, UNetV2_3, UNetV2_2
from lib.models.spconv_utils import replace_feature, spconv

class spcenterDet(nn.Module):
    def __init__(self, heads, image_size = [512,512], img_num = 20, layers = 4, thresh=None):
        super().__init__()
        self.thresh=thresh
        input_channels = 4
        head_conv=128
        grid_size = np.array([image_size[0], image_size[1], img_num - 1])
        self.points_all = img_num*image_size[0]*image_size[1]
        if  layers==4:
            self.sp_backbone = UNetV2(input_channels, grid_size)
        elif layers==3:
            self.sp_backbone = UNetV2_3(input_channels, grid_size)
        elif layers == 2:
            self.sp_backbone = UNetV2_2(input_channels, grid_size)
        else:
            raise Exception('Not a valid mode!!!!!')
        head_input_channel = self.sp_backbone.num_point_features
        ###get head conv
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            name_1 = 'subm1'+head
            name_2 = 'subm2'+head
            if head_conv > 0:
                if 'hm' in head:
                    fc = spconv.SparseSequential(
                            spconv.SubMConv3d(head_input_channel, head_conv, 3, padding=1, bias=False, indice_key=name_1),
                        nn.ReLU(),
                        spconv.SubMConv3d(head_conv, classes, 3, padding=1, bias=True, indice_key=name_2),
                        )
                else:
                    fc = spconv.SparseSequential(
                        spconv.SubMConv3d(head_input_channel, head_conv, 3, padding=1, bias=False, indice_key=name_1),
                        nn.ReLU(),
                        spconv.SubMConv3d(head_conv, classes, 3, padding=1, bias=False, indice_key=name_2),
                        )
            else:
                fc = spconv.SubMConv3d(head_input_channel, classes, 3, padding=1, bias=True, indice_key=name_1)
            ###
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            self.__setattr__(head, fc)

            self.sigmoid = nn.Sigmoid()

            self.tau = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.tau.data.fill_(1)
            self.conv_std = nn.Sequential(
                nn.AdaptiveAvgPool2d([1, 1]),
                nn.Conv2d(img_num, img_num, 1),
                nn.ReLU(inplace=True)
            )

            self.relu = nn.ReLU(inplace=True)

    def preprocess(self, img, img_gray):
        mask_all = torch.zeros_like(img_gray)
        diff = img_gray - torch.median(img_gray[:,:,::3], 2)[0].unsqueeze(2)
        diff = abs(diff)
        #####
        diff0 = diff.clone()
        std = torch.std(diff, [-2, -1]).unsqueeze(-1).unsqueeze(-1)
        mean = torch.mean(diff, [-2, -1]).unsqueeze(-1).unsqueeze(-1)
        if self.thresh is not None:
            lr_th = mean + self.thresh * std
        else:
            lr_th = mean + 3 * std
        diff = self.relu(diff-lr_th)
        coords = torch.nonzero(diff.squeeze(1))
        img1 = torch.cat([img, diff], 1)
        features = img1[coords[:,0],:,coords[:,1], coords[:,2], coords[:,3]]
        # print(features.shape[0]/1024/1024/20*100)
        coords = coords.contiguous()
        batch_dict = {}
        batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = coords.to(features.device)
        batch_dict['batch_size'] = img_gray.shape[0]
        del img, img_gray
        return batch_dict, diff0, mask_all

    def forward(self, batch):
        # print(self.tau1)
        _, _, _, h, w = batch['input'].shape
        batch_dict, diff0, mask_all = self.preprocess(batch['input'], batch['input_gray'])
        sp_backbone_out = self.sp_backbone(batch_dict)
        z = {}
        for head in self.heads:
            input_sp_tensor = sp_backbone_out['encoded_spconv_tensor']
            out_h = self.__getattr__(head)(input_sp_tensor)

            if 'hm' in head:
                out_h = replace_feature(out_h, self.sigmoid(out_h.features))
                spatial_features = out_h.dense()
                spatial_features = torch.clamp(spatial_features, min=1e-4, max=1 - 1e-4)
            else:
                spatial_features = out_h.dense()
            z[head] = spatial_features
        z['mask_all'] = diff0
        z['voxel_coords'] = batch_dict['voxel_coords']
        z['lasso'] = torch.sum(mask_all, dim=[-1,-2]) / (h * w)
        return [z]

def sp_centerDet_minus(heads, image_size = [512,512], img_num = 20, layers=4, thresh=None):
    model = spcenterDet(heads,  image_size = image_size, img_num = img_num, layers=layers, thresh=thresh)
    return model
