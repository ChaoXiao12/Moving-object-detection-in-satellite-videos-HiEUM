from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from lib.utils1.image import flip, color_aug
from lib.utils1.image import get_affine_transform, affine_transform
from lib.utils1.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils1.image import draw_dense_reg
import math
from lib.utils1.opts import opts

from lib.utils1.augmentations import Augmentation

import torch.utils.data as data
from collections import defaultdict


class COCO_rs_car(data.Dataset):
    opt = opts().parse()
    num_classes = 1
    default_resolution = [512,512]
    mean = np.array([0.49965, 0.49965, 0.49965],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255],
                   dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, opt, split):
        super(COCO_rs_car, self).__init__()
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.img_dir0 = self.opt.data_dir

        self.img_dir = self.opt.data_dir

        if split == 'train':
            self.resolution = [320, 320]
            self.resolution_ori = [512, 512]
            self.annot_path = os.path.join(
                self.img_dir0, 'annotations',
                'train_mot.json')
        else:
            self.resolution = [1024, 1024]
            self.annot_path = os.path.join(
                self.img_dir0, 'annotations',
                'test1024_mot.json')

        self.down_ratio = opt.down_ratio
        self.max_objs = opt.K
        self.seqLen = opt.seqLen

        self.class_name = [
            '__background__', 'car']
        self._valid_ids = [
            1, 2]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}  # 生成对应的category dict

        self.split = split
        self.opt = opt

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))
        self.video_to_images = defaultdict(list)
        count=0
        for image in self.coco.dataset['images']:
            self.video_to_images[image['video_id']].append([self.images[count],image])
            count+=1

        if(split=='train'):
            self.aug = Augmentation()
        else:
            self.aug = None

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    # 遍历每一个标注文件解析写入detections. 输出结果使用
    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir, time_str):
        json.dump(self.convert_eval_format(results),
                  open('{}/results_{}.json'.format(save_dir,time_str), 'w'))

        print('{}/results_{}.json'.format(save_dir,time_str))

    def run_eval(self, results, save_dir, time_str):
        self.save_results(results, save_dir, time_str)
        coco_dets = self.coco.loadRes('{}/results_{}.json'.format(save_dir, time_str))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        precisions = coco_eval.eval['precision']

        return stats, precisions

    def run_eval_just(self, save_dir, time_str, iouth):
        coco_dets = self.coco.loadRes('{}/{}'.format(save_dir, time_str))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox", iouth = iouth)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_5 = coco_eval.stats
        precisions = coco_eval.eval['precision']

        return stats_5, precisions

    def _coco_box_to_bbox(self, box):
        # if box[2]<30 and box[3]<30:
        #     box[2] = box[0] + box[2]
        #     box[3] = box[1] + box[3]
        # if box[2]<30 and box[3]<30:
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        bbox = np.array(box,
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

