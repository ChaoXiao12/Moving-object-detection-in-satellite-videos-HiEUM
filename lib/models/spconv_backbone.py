from functools import partial

import torch.nn as nn

from lib.models.spconv_utils import replace_feature, spconv

import torch


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, algo = None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key,algo=algo)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key,algo=algo)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False,algo=algo)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, algo=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None

        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key,algo=algo
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key,algo=algo
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, input_channels, grid_size, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # pool = spconv.pool.SparseMaxPool()

        algo = spconv.ConvAlgo.Native

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1',algo=algo),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block # 一个block就是一个conv+bn+relu，通过‘indice_key’确定是spconv还是subm

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1',algo=algo),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21] 一般情况是一个spconv跟两个subm层，公共构成了一个sp卷积单位（类比一个卷积层）
            block(16, 32, 3, norm_fn=norm_fn, stride=[2,2,2], padding=1, indice_key='spconv2', conv_type='spconv',algo=algo),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2',algo=algo),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2',algo=algo),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=[2,2,2], padding=1, indice_key='spconv3', conv_type='spconv',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',algo=algo),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=[2,2,2], padding=(1, 1, 1), indice_key='spconv4', conv_type='spconv',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',algo=algo),
        )

        self.upconv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, padding=(1, 1, 1), indice_key='spconvup4', conv_type='spconv',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submup4',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submup4',algo=algo),
        )
        self.up4 = spconv.SparseInverseConv3d(64, 64, 3, indice_key='spconv4', bias=False,algo=algo)
        self.upconv3 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(128, 64, 3, norm_fn=norm_fn, padding=(1, 1, 1), indice_key='spconvup3', conv_type='spconv',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submup3',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submup3',algo=algo),
        )
        self.up3 = spconv.SparseInverseConv3d(64, 32, 3, indice_key='spconv3', bias=False, algo=algo)
        self.upconv2 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 32, 3, norm_fn=norm_fn, padding=(1, 1, 1), indice_key='spconvup2', conv_type='spconv',algo=algo),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submup2',algo=algo),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submup2',algo=algo),
        )
        self.up2 = spconv.SparseInverseConv3d(32, 16, 3, indice_key='spconv2', bias=False,algo=algo)
        self.upconv1 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(32, 16, 3, norm_fn=norm_fn, padding=(1, 1, 1), indice_key='spconvup1', conv_type='spconv',algo=algo),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='submup1',algo=algo),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='submup1',algo=algo),
        )
        # self.up1 = spconv.SparseInverseConv3d(16, 16, 3, indice_key='spconv1', bias=False)
        # self.conv_out_debug = spconv.SparseSequential(
        #     # [200, 150, 2] -> [200, 150, 1]
        #     spconv.SparseConv3d(128, 128, 3, stride=(2, 1, 1), padding=1,
        #                         bias=False, indice_key='spconv_down_debug'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )

        last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(16, 1, 3, stride=(1, 1, 1), padding=1,
        #                         bias=False, indice_key='spconv_down2',algo=algo),
        #     # norm_fn(1),
        #     # nn.ReLU(),
        #     # spconv.SparseConv3d(16, 1, 3, stride=(1, 1, 1), padding=1,
        #     #                     bias=False, indice_key='spconv_down3'),
        # )
        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(16, 16, 3, padding=1, bias=False, indice_key='submout',algo=algo),
            norm_fn(16),
            nn.ReLU(),
        )
        self.out_channel = 16
        self.num_point_features = self.out_channel
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64,
            'x_out': 128,   ######
            'bev': 256,
            'bev_2d': 512,
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        ## 构建sp_tensor, 最主要的是voxel_features和voxel_coords,
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,          #(N, 4) float32,  eg: torch.Size([64000, C])
            indices=voxel_coords.int(),       #(N, 4)[bs_idx, z, y, x], int32， eg: torch.Size([64000, 4])
            spatial_shape=self.sparse_shape,  #（3）  z, y, x 的容量， eg: array([  41, 1600, 1408])
            batch_size=batch_size             # bs  eg: 4
        )
        ## 创建好之后里面包括了features， indices， spatial_shape 等成分，可以直接用input_sp_tensor.features 方式调用
        
        # 后面的网络创建基本就照葫芦画瓢了，注意一点无论stride还是卷积核的三元组都是（维度3，维度2，维度1），是建议将时间维度T放在开头；单个整数默认是三元组内部元素相同

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_4 = self.upconv4(x_conv4)
        x_up_4 = self.up4(x_4)

        x_conv3 = self.replace_feature(x_conv3, torch.cat((x_conv3.features, x_up_4.features), dim=1))
        x_3 = self.upconv3(x_conv3)
        x_up_3 = self.up3(x_3)

        x_conv2 = self.replace_feature(x_conv2, torch.cat((x_conv2.features, x_up_3.features), dim=1))
        x_2 = self.upconv2(x_conv2)
        x_up_2 = self.up2(x_2)

        x_conv1 = self.replace_feature(x_conv1, torch.cat((x_conv1.features, x_up_2.features), dim=1))
        x_1 = self.upconv1(x_conv1)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_1)

        # out_debug = self.conv_out_debug(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1, #[41, 1600, 1408]; ([240884, 16])
                'x_conv2': x_conv2, #[21, 800, 704];   ([401179, 32])
                'x_conv3': x_conv3, #[11, 400, 352];   ([261878, 64])
                'x_conv4': x_conv4, #[5, 200, 176];    ([115506, 64])
                'x_out': out,   #####[2, 200, 176];    ([94886, 128])                          encoded_spconv_tensor
                # 'out_debug': out_debug #[1, 200, 176]; ([16894, 128])
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_out': [8, 8, 20],   ######
                'out_debug': [8, 8, 40], 
                'bev': 8, ######
                'bev_2d': 8, ######
            }
        })

        return batch_dict

    def replace_feature(self, out, new_features):
        if "replace_feature" in out.__dir__():
            # spconv 2.x behaviour
            return out.replace_feature(new_features)
        else:
            out.features = new_features
            return out


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
