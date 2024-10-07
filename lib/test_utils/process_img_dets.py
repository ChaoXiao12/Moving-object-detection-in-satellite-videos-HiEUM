from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils1.decode import ctdet_decode
from lib.utils1.post_process import ctdet_post_process
import cv2

from lib.external1.nms import soft_nms
from lib.test_utils.get_coords import *

def pre_process(image, scale=1):
    height, width = image.shape[2:4]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height,
            'out_width': inp_width}
    return meta

def preprocess(img_list, dataset):
    seq_num = len(img_list)
    img = np.zeros([dataset.resolution[0], dataset.resolution[1], 3, seq_num])
    imgs = np.zeros([dataset.resolution[0], dataset.resolution[1], 3, seq_num])
    imgs_gray = np.zeros([dataset.resolution[0], dataset.resolution[1], 1, seq_num])
    a1 = time.time()
    for ii in range(seq_num):
        img_id_cur = img_list[ii]
        im = cv2.imread(img_id_cur)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img[:, :, :, ii] = im
        ###
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        imgs_gray[:, :, 0, ii] = im_gray
        # normalize
        inp_i = (im.astype(np.float32) / 255.)
        inp_i = (inp_i - dataset.mean) / dataset.std
        imgs[:, :, :, ii] = inp_i

    a2 = time.time()
    inp = np.expand_dims(imgs.transpose(2, 3, 0, 1).astype(np.float32),0)
    inp_gray = np.expand_dims(imgs_gray.transpose(2, 3, 0, 1).astype(np.float32),0)
    # 读入图像
    meta = pre_process(inp, 1)
    # batch_dict = get_points(inp, inp_gray)
    batch_dict ={}
    input_imgs = {}
    input_imgs['input'] = inp
    input_imgs['input_gray'] = inp_gray
    return batch_dict, meta, img, input_imgs

def process(model, image, return_time, opt, K=128):
    with torch.no_grad():
        output = model(image)[-1]
        hm = output['hm']
        wh = output['wh']
        if opt.off_flag:
            reg = output['reg']
        else:
            reg = None
        torch.cuda.synchronize()
        forward_time = time.time()

        if reg is not None:
            dets =  ctdet_decode(hm[0].transpose(0,1), wh[0].transpose(0,1),
                    reg=reg[0].transpose(0,1), K=K)
        else:
            dets =  ctdet_decode(hm[0].transpose(0,1), wh[0].transpose(0,1),
                       reg=None, K=K)
    if return_time:
        return output, dets, forward_time
    else:
        return output, dets

def post_process(dets_all, meta, num_classes=1, scale=1, max_per_image=100):
    # 后处理
    rets = []
    dets_post = []
    dets_all = dets_all.unsqueeze(1).detach().cpu().numpy()
    for iii in range(dets_all.shape[0]):
        dets = dets_all[iii]
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], num_classes)
        for j in range(1, num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        detection = []
        det = dets[0]
        dets_post.append(det)
        detection.append(det)
        ret = merge_outputs(detection, num_classes, max_per_image)
        rets.append(ret)
    return rets, dets_post

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.1, method=1)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results








