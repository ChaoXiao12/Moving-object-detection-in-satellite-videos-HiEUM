from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from lib.external1.nms import soft_nms
from skimage import measure
from lib.utils1.sort import *

def get_results_unfold(output):
    B = output['B']
    T = output['T']
    N = output['N']
    torch.cuda.synchronize()
    forward_time = time.time()
    ####
    mean = torch.mean(T, dim=[-1, -2]).unsqueeze(-1).unsqueeze(-1)
    std = torch.std(T, dim=[-1, -2]).unsqueeze(-1).unsqueeze(-1)
    ######
    T = T - mean - std * 10
    T[T < 0] = 0

    mask1 = torch.zeros_like(T)
    mask1[T > 0] = 1
    output['mask'] = mask1
    output['T'] = T

    mask1 = mask1.cpu().numpy()
    mask = T.cpu().numpy()

    b, c, t, h, w = mask.shape

    dets = []
    for i in range(t):
        seg = mask1[0, 0, i]
        im = mask[0, 0, i]
        topk_coords, coords = get_det_result_from_im(seg, im)
        dets.append(topk_coords)

    return dets, output

def get_points(input, input_gray, input_hm=None):

    if type(input) is not np.ndarray:
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

def get_det_result_from_im(seg, image_out):
    top_k = 80
    area_th_min = 2
    area_th_max = 110
    image = measure.label(seg, connectivity=2)
    prop_regions = measure.regionprops(image, intensity_image=image_out)
    coords = np.array([[list(i.bbox) + [i.intensity_max]] for i in prop_regions if (i.area > area_th_min and i.area < area_th_max)])
    # coords = np.array([i.bbox for i in prop_regions])
    coords = coords.reshape(-1, 5)
    coords1 = coords.copy()
    coords1[:, 0] = coords[:, 1]
    coords1[:, 1] = coords[:, 0]
    coords1[:, 2] = coords[:, 3]
    coords1[:, 3] = coords[:, 2]

    results = {}
    results[1] = coords1.astype(np.float32)
    # nms
    soft_nms(results[1], Nt=0.5, method=2)
    # get top_k detections
    scores = results[1][:, -1]
    if len(scores) > top_k:
        kth = len(scores) - top_k
        thresh = np.partition(scores, kth)[kth]
        keep_inds = (results[1][:, 4] >= thresh)
        results[1] = results[1][keep_inds]

    return results[1], coords1

def filt_coords_by_trajs( dets_all_ori, max_age=30, min_hits=1, iou_threshold=0.1):
    ######
    mot_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    ids = []
    trajs = []
    ######
    for i_im in range(len(dets_all_ori)):
        det_coords = dets_all_ori[i_im]
        # det_coords[:,:2]-=2
        # det_coords[:, 2:4] += 2
        track_bbs_ids = mot_tracker.update(det_coords)
        for it in range(track_bbs_ids.shape[0]):
            id = track_bbs_ids[it, -1]
            coord = track_bbs_ids[it, :4]
            if id not in ids:
                ids.append(track_bbs_ids[it, -1])
                trajs.append([])
            index = ids.index(id)
            trajs[index].append([i_im] + coord.tolist())

    trajs_filt = []
    for traj_i in trajs:
        if len(traj_i) < 15:
            continue
        a = np.array(traj_i)
        ct = (a[:, 3:5] + a[:, 1:3]) / 2
        d = ct[1:, :] - ct[:-1, :]
        d = (d[:, 0] ** 2 + d[:, 1] ** 2) ** 0.5
        v = d / (a[1:, 0] - a[:-1, 0])
        v_mean = abs(v).mean()
        if v_mean < 0.55 and len(traj_i) < 30:
            continue
        trajs_filt.append(traj_i)

    det_for_images = [[] for i in range(len(dets_all_ori))]
    images = [i for i in range(len(dets_all_ori))]
    count = 0
    for i_traj in trajs_filt:
        count = count + 1
        for i_trajkk in i_traj:
            index = images.index(i_trajkk[0])
            det_for_images[index].append(i_trajkk[1:] + [count])
    return trajs, trajs_filt, det_for_images