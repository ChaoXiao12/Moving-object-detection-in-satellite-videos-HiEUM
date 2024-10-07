import cv2
import torch
from skimage import measure
import os
import numpy as np

from PIL import Image

from lib.external1.nms import soft_nms

from lib.utils1.sort import *
from lib.LRSD.WSNMSTIPT_dp_without_B_norm import WSNMSTIPT_dp_without_B_norm

def preprocess(img_list, resolution=[512,512]):
    seq_num = len(img_list)
    imgs_gray = np.zeros([resolution[0], resolution[1],  seq_num])
    a1 = time.time()
    for ii in range(seq_num):
        img_id_cur = img_list[ii]
        im = cv2.imread(img_id_cur)
        ###
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        imgs_gray[:, :, ii] = im_gray
    return imgs_gray

def get_tar_ims(data_dir, save_tar_ims=None,  th_std = 5):
    test_upper_path = os.path.join(data_dir,'images/train/')
    data_folder_list = os.listdir(test_upper_path)
    data_folder_list.sort()
    if save_tar_ims is None:
        save_tar_ims = os.path.join(data_dir, 'lrsd', 'train')
    if not os.path.exists(save_tar_ims):
        os.makedirs(save_tar_ims)
    #params
    patch_len = 16
    Lambda = 1
    mu = 5e-4
    beta = 100
    rho = 1.5
    for ii in range(0, len(data_folder_list)):
        data_folder_path = os.path.join(test_upper_path, data_folder_list[ii], 'img1')
        save_coords_folder = os.path.join(save_tar_ims, data_folder_list[ii],'coords_unfilt')
        if not os.path.exists(save_coords_folder):
            os.makedirs(save_coords_folder)
        img_list = os.listdir(data_folder_path)
        img_list = [i for i in img_list if i.endswith('.jpg')]
        img_list.sort()
        imgs_number = len(img_list)
        overlap_flag = 0
        if len(img_list)%patch_len==0:
            patch_num = len(img_list)//patch_len
        else:
            patch_num = len(img_list) // patch_len+1
            overlap_flag=1
        for pk in range(0, patch_num):
            time_start = time.time()
            if overlap_flag and pk==patch_num-1:
                patch_ids = [i for i in range(imgs_number-patch_len, imgs_number)]
                patch_ims = img_list[imgs_number-patch_len:imgs_number]
            else:
                patch_ids = [i for i in range(pk*patch_len, (pk+1)*patch_len)]
                patch_ims = img_list[pk*patch_len : (pk+1)*patch_len]
            patch_ims_path = [os.path.join(data_folder_path, i) for i in patch_ims]
            input_imgs = preprocess(patch_ims_path)
            input_imgs_t = torch.from_numpy(input_imgs).cuda()
            B_hat_t = torch.median(input_imgs_t, 2)[0].unsqueeze(2).repeat(1,1, input_imgs_t.shape[2])
            out = WSNMSTIPT_dp_without_B_norm(input_imgs_t, B_hat_t, Lambda, mu, beta, rho)
            tar_ims = out['T'].cpu().numpy()
            for ik in range(len(patch_ims)):
                im = tar_ims[:,:,ik]
                mask = np.zeros_like(im)
                mask[im > im.mean() + th_std * im.std()] = 1
                topk_coords, coords = get_det_result_from_im(mask, im)
                txt_save_name_ori = os.path.join(save_coords_folder,patch_ims[ik].replace('.jpg', '.txt'))
                fid_txt_ori = open(txt_save_name_ori, 'w')
                #####no filt
                for da in range(topk_coords.shape[0]):
                    coord_da = topk_coords[da]
                    fid_txt_ori.write('%d\t%d\t%d\t%d\t0\t%d\n' % (
                    coord_da[0], coord_da[1], coord_da[2], coord_da[3], coord_da[4]))
                fid_txt_ori.close()
        print('folder', data_folder_list[ii], 'get lrsd results done!!!')

def get_mask(dets, img_size):
    mask = np.zeros(img_size[0],img_size[1])
    for i in range(dets.shape[0]):
        mask[dets[i,0]:dets[i,2], dets[i,1]:dets[i,3]] = 1
    return mask

def get_det_result_from_im(seg, image_out):
    top_k = 80
    area_th_min = 4
    area_th_max = 80
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

def generate_labels(data_dir=None):
    if data_dir is None:
        data_dir = './datasets/RsCarData'
    save_path_upper = os.path.join(data_dir, 'lrsd/train/')
    if not os.path.exists(save_path_upper):
        os.makedirs(save_path_upper)
    #get tar ims from lrsd
    th_std = 5
    get_tar_ims(data_dir, save_path_upper, th_std)
    data_list = os.listdir(save_path_upper)
    data_list.sort()
    for dfk in range(0, len(data_list)):
        data_f = data_list[dfk]
        data_folder = os.path.join(save_path_upper, data_f, 'coords_unfilt')
        #####
        save_folder = os.path.join(save_path_upper, data_f)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_txt_folder = os.path.join(save_folder, 'coords_update')
        if not os.path.exists(save_txt_folder):
            os.mkdir(save_txt_folder)
        save_txt_folder_ori = os.path.join(save_folder, 'coords_filt')
        if not os.path.exists(save_txt_folder_ori):
            os.mkdir(save_txt_folder_ori)
        ######
        im_list = os.listdir(data_folder)
        im_list = [i for i in im_list if i.endswith('.txt')]
        im_list.sort()
        ######
        mot_tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.1)
        ids = []
        trajs = []
        ######
        dets_all_ori = []
        for i_im in range(0, len(im_list)):
            im_name = os.path.join(im_list[i_im])
            im_path = os.path.join(data_folder, im_name)
            det_coords = np.loadtxt(im_path).reshape(-1,6)

            dets_all_ori.append(det_coords)
            track_bbs_ids = mot_tracker.update(det_coords)
            for it in range(track_bbs_ids.shape[0]):
                id = track_bbs_ids[it,-1]
                coord = track_bbs_ids[it,:4]
                if id not in ids:
                    ids.append(track_bbs_ids[it,-1])
                    trajs.append([])
                index = ids.index(id)
                trajs[index].append([i_im]+coord.tolist())

        trajs_filt = []
        for traj_i in trajs:
            if len(traj_i)<15:
                continue
            a = np.array(traj_i)
            ct = (a[:,3:5]+a[:,1:3])/2
            d = ct[1:, :]-ct[:-1,:]
            d = (d[:,0]**2+d[:,1]**2)**0.5
            v = d/(a[1:,0]-a[:-1,0])
            v_mean = abs(v).mean()
            # print(v_mean)
            if v_mean<0.55:
                continue
            trajs_filt.append(traj_i)

        det_for_images = [[] for i in range(len(im_list))]
        images = [i for i in range(len(im_list))]
        count = 0
        for i_traj in trajs_filt:
            count=count+1
            for i_trajkk in i_traj:
                index = images.index(i_trajkk[0])
                det_for_images[index].append(i_trajkk[1:]+[count])
        ##################

        for kk in range(len(det_for_images)):
            #######
            txt_save_name = os.path.join(save_txt_folder, im_list[kk].replace('.jpg', '.txt'))
            fid_txt = open(txt_save_name, 'w')
            #
            txt_save_name_ori = os.path.join(save_txt_folder_ori, im_list[kk].replace('.jpg', '.txt'))
            fid_txt_ori = open(txt_save_name_ori, 'w')
            #####update
            for coord in det_for_images[kk]:
                fid_txt.write('%d\t%d\t%d\t%d\t0\t%d\n'%(coord[0], coord[1], coord[2], coord[3], coord[4]))
            #####filt
            for coord11 in det_for_images[kk]:
                fid_txt_ori.write('%d\t%d\t%d\t%d\t0\t%d\n'%(coord11[0], coord11[1], coord11[2], coord11[3], coord11[4]))
            fid_txt.close()
            fid_txt_ori.close()
        print('folder', data_f, 'get filtered lrsd results done!!!')


if __name__ == '__main__':
    data_dir = '/media/wellwork/L/xc/datasets/RsCarData'
    generate_labels(data_dir)





