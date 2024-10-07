from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from lib.loss.losses import FocalLoss
from lib.loss.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from lib.utils1.decode import ctdet_decode
from lib.utils1.utils import _sigmoid
from lib.utils1.debugger import Debugger
from lib.utils1.post_process import ctdet_post_process
from lib.Trainer.base_trainer_points import BaseTrainer
import cv2


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()  # torch.nn.MSELoss()
        # self.crit = torch.nn.MSELoss()
        self.crit_reg = RegL1Loss()  # RegLoss()
        self.crit_wh = torch.nn.L1Loss(reduction='sum')  # NormRegL1Loss() # RegWeightedL1Loss()
        self.opt = opt
        self.wh_weight = opt.wh_weight
        self.hm_weight = opt.hm_weight
        self.off_weight = opt.off_weight
        self.num_stacks = 1

    def forward(self, outputs, batch):
        hm_loss, wh_loss, off_loss, lasso_loss = 0, 0, 0, 0

        output = outputs[0]

        b,c,t,h,w = output['hm'].shape

        for it in range(t):
            if self.opt.hm_flag:
                hm_loss += self.crit(output['hm'][:,:,it].contiguous(), batch['hm'][:,:,it]) / self.num_stacks

            if self.opt.wh_flag:
                wh_loss += self.crit_reg(
                    output['wh'][:,:,it].contiguous(), batch['reg_mask'][:, it],
                    batch['ind'][:, it], batch['wh'][:,it]) / self.num_stacks

            if self.opt.off_flag:
                off_loss += self.crit_reg(output['reg'][:,:, it].contiguous(), batch['reg_mask'][:,it],
                                          batch['ind'][:,it], batch['reg'][:,it])

        hm_loss = hm_loss/t
        wh_loss = wh_loss/t
        off_loss = off_loss/t

        loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + \
               self.off_weight * off_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}

        return loss, loss_stats


class CtdetTrainer_points(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer_points, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        if opt.off_flag:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        else:
            loss_states = ['loss', 'hm_loss', 'wh_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]