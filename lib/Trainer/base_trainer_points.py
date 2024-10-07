from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from lib.utils1.data_parallel import DataParallel
from lib.utils1.utils import AverageMeter
from lib.utils1.decode import ctdet_decode
from lib.utils1.post_process import ctdet_post_process
import numpy as np
from lib.external1.nms import soft_nms

from lib.test_utils.test import test
from lib.test_utils.test_update import test_update

def post_process(output, meta, num_classes=1, scale=1):
    # decode
    # hm = output['hm'].sigmoid_()
    # wh = output['wh']
    # reg = output['reg']

    hm = output['hm'][:,:,0].transpose(-1,-2).contiguous()
    wh = output['wh'][:,:,0].transpose(-1,-2).contiguous()
    reg = output['reg'][:,:,0].transpose(-1,-2).contiguous()

    torch.cuda.synchronize()
    dets = ctdet_decode(hm, wh, reg=reg)
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.5, method=2)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # print(batch['input'].shape)
        # outputs = self.model(batch['batch_dict'])
        outputs = self.model(batch)
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        self.model = model

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def get_points(self, input, input_gray, input_hm=None):

        input = input.numpy()
        input_gray = input_gray.numpy()

        b,c,img_num,h,w = input.shape

        coords_all = []
        features_all = []

        for ib in range(b):
            img_rgb_t = input[ib, :]
            imgt = input_gray[ib,:]
            imgt = imgt[0]
            bt = np.expand_dims(np.median(imgt, 0), 0)
            dt = imgt - bt
            maskt = np.zeros_like(dt)
            a = dt.reshape([img_num, -1])
            th = np.expand_dims(np.mean(a, axis=1) + 3 * np.std(a, axis=1), [-2, -1])
            maskt[dt > th] = 1
            if ib==b-1:
                maskt[-1,-1]=1
            xx = [i for i in range(imgt.shape[1])]
            yy = [i for i in range(imgt.shape[2])]
            zz = [i for i in range(img_num)]
            grid0 = np.meshgrid(xx, yy, zz)
            grid1 = np.array([grid0[1], grid0[0], grid0[2]])
            grid1 = grid1.transpose(1, 2, 3, 0)
            maskt = maskt.transpose(1, 2, 0)
            img_rgb_t = img_rgb_t.transpose(2, 3, 1, 0)
            coords = grid1[maskt > 0, :]
            features = img_rgb_t[maskt > 0, :].astype(np.float32)
            # ###check
            # print((img_rgb_t[coords[:, 0], coords[:, 1], coords[:, 2], :] - features).sum())
            coords_out = np.zeros([coords.shape[0],4])
            coords_out[:,:1] = ib
            # coords_out[:, 1:] = coords[:, ]
            for iiii in range(3):
                coords_out[:, iiii+1] = coords[:,2-iiii]
            coords_all.append(torch.from_numpy(coords_out))
            features_all.append(torch.from_numpy(features))

        batch_dict = {}
        batch_dict['voxel_features'] = torch.cat(features_all, 0)
        batch_dict['voxel_coords'] = torch.cat(coords_all, 0)
        batch_dict['batch_size'] = b

        return batch_dict

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)//self.opt.data_sampling   #20
        # bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
              break
            data_time.update(time.time() - end)

            # time_pre = time.time()

            # for k in batch:
            #     if k != 'meta' and k != 'file_name' and k!='input_gray' and k!='im_ids':
            #         if k=='input':
            #             batch_dict = self.get_points(batch['input'], batch['input_gray'], batch['hm'])
            #             for kkk, vvv in batch_dict.items():
            #                 if kkk == 'batch_size' or kkk=='im_ids':
            #                     continue
            #                 batch_dict[kkk] = vvv.to(device=opt.device, non_blocking=True)
            #         else:
            #             batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            # for k in batch:
            #     if k != 'meta' and k != 'file_name' and k!='input_gray' and k!='im_ids':

            for k in batch:
                if k != 'meta' and k != 'file_name'  and k!='im_ids':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            # batch_dict, mask_all = self.model.preprocess(batch['input'].to(opt.device),
            #                               batch['input_gray'].to(opt.device))
            # ####
            # batch['batch_dict'] = batch_dict
            ####
            # time_pre1 = time.time()
            # print('time_preprocessd:', time_pre1-time_pre)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)

            print_str = 'phase=%s, epoch=%5d, iters=%d/%d, time=%0.4f' \
                  % (phase, epoch,iter_id+1,num_iters, time.time() - end,
                     )

            for k,v in loss_stats.items():
                if k in self.loss_stats:
                    stri = ', %s=%0.4f'%(k,v.mean().cpu().detach().numpy())
                    print_str = print_str + stri

            print(print_str)

            # print('phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f' \
            #       % (phase, epoch,iter_id+1,num_iters, time.time() - end,
            #          loss.mean().cpu().detach().numpy(),
            #          loss_stats['hm_loss'].mean().cpu().detach().numpy(),
            #          loss_stats['wh_loss'].mean().cpu().detach().numpy(),
            #          loss_stats['off_loss'].mean().cpu().detach().numpy()))

            # print(
            #     'phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f' \
            #     % (phase, epoch, iter_id + 1, num_iters, time.time() - end,
            #        loss.mean().cpu().detach().numpy(),
            #        loss_stats['hm_loss'].mean().cpu().detach().numpy())
            # )

            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = 1 / 60.

        return ret, results

    def run_eval_points(self, phase, epoch, data_loader, base_s, dataset, modelPath, model_path_name):

        ret = test(self.opt, phase, modelPath, show_flag=False, results_name=model_path_name+'results_latest', save_mat=True)
        results=[]
        stats1 = []
        return ret, results, stats1

    def run_eval_epoch(self, phase, epoch, data_loader, base_s, dataset):
        model_with_loss = self.model_with_loss

        if len(self.opt.gpus) > 1:
            model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        end = time.time()

        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
              break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta' and k != 'file_name' and k!='input_gray' and k!='im_ids':
                    if k=='input':
                        batch_dict = self.get_points(batch['input'], batch['input_gray'], batch['hm'])
                        for kkk, vvv in batch_dict.items():
                            if kkk == 'batch_size' or kkk=='im_ids':
                                continue
                            batch_dict[kkk] = vvv.to(device=opt.device, non_blocking=True)
                    else:
                        batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            ####
            batch['batch_dict'] = batch_dict
            ####
            output, loss, loss_stats = model_with_loss(batch)

            inp_height, inp_width = batch['input'].shape[3],batch['input'].shape[4]
            c = np.array([inp_width / 2., inp_height / 2.], dtype=np.float32)
            s = max(inp_height, inp_width) * 1.0

            meta = {'c': c, 's': s,
                    'out_height': inp_height,
                    'out_width': inp_width}

            dets = post_process(output, meta)
            ret = merge_outputs([dets], num_classes=1, max_per_image=opt.K)
            results[im_id.numpy().astype(np.int32)[0]] = ret

            loss = loss.mean()
            batch_time.update(time.time() - end)

            print('phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f' \
                  % (phase, epoch,iter_id+1,num_iters, time.time() - end,
                     loss.mean().cpu().detach().numpy(),
                     loss_stats['hm_loss'].mean().cpu().detach().numpy(),
                     loss_stats['wh_loss'].mean().cpu().detach().numpy(),
                     loss_stats['off_loss'].mean().cpu().detach().numpy()))
            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        stats1, _ = dataset.run_eval(results, opt.save_results_dir, 'latest')
        ret['time'] = 1 / 60.
        ret['ap50'] = stats1[1]

        return ret, results, stats1

    def update_label(self, epoch, data_loader, base_s, dataset, modelPath):
        phase = 'train'
        test_update(self.opt, phase, modelPath, show_flag=False, results_name='results_latest', epoch=epoch)

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader, base_s, dataset, modelPath, model_path_name):
        # return self.run_epoch('val', epoch, data_loader)

        return self.run_eval_points('test', epoch, data_loader, base_s, dataset, modelPath,model_path_name)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)