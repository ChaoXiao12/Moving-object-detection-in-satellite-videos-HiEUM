from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.dataset_factory import get_dataset

from lib.test_utils.process_img_dets import *

def test_update(opt, split, modelPath, show_flag, results_name, save_mat=False, epoch=0):

    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    opt.test_large_size = True

    # Logger(opt)
    print(opt.model_name)

    dataset = get_dataset(opt)

    DataVal = dataset(opt, 'train')
    if opt.off_flag:
        head = {'hm': DataVal.num_classes, 'wh': 2, 'reg': 2}
    else:
        head = {'hm': DataVal.num_classes, 'wh': 2}
    DataVal.resolution = [512,512]
    model = get_det_net(head, opt.model_name, DataVal.resolution, opt.seqLen, opt)  # 建立模型
    model = load_model(model, modelPath)
    model = model.to(opt.device)
    model.eval()

    conf_filtered = opt.conf_filtered

    return_time = False
    num_classes = dataset.num_classes
    max_per_image = opt.K

    test_upper_path = opt.data_dir + 'images/train/'
    data_folder_list = os.listdir(test_upper_path)
    patch_len = opt.seqLen

    save_mat_path_upper = test_upper_path.replace('images', 'lrsd')

    for ii in range(len(data_folder_list)):
        data_folder_path = os.path.join(test_upper_path, data_folder_list[ii], 'img1')
        save_txt_update_folder = os.path.join(save_mat_path_upper, data_folder_list[ii], 'coords_update')
        # save_txt_update_folder_epoch = os.path.join(save_mat_path_upper, data_folder_list[ii], 'coords_update_%d'%epoch)
        if not os.path.exists(save_txt_update_folder):
            os.mkdir(save_txt_update_folder)
        txt_coords_ori_folder = os.path.join(save_mat_path_upper, data_folder_list[ii], 'coords_update')
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
        detections_all = [[] for i in range(len(img_list))]
        detections_ori = [[] for i in range(len(img_list))]
        for ii_m in range(len(img_list)):
            txt_path = os.path.join(txt_coords_ori_folder, img_list[ii_m].replace('.jpg', '.txt'))
            coord_ori = np.loadtxt(txt_path)
            detections_ori[ii_m] = coord_ori.reshape(-1,6)
        for pk in range(patch_num):
            time_start = time.time()
            if overlap_flag and pk==patch_num-1:
                patch_ids = [i for i in range(imgs_number-patch_len, imgs_number)]
                patch_ims = img_list[imgs_number-patch_len:imgs_number]
            else:
                patch_ids = [i for i in range(pk*patch_len, (pk+1)*patch_len)]
                patch_ims = img_list[pk*patch_len : (pk+1)*patch_len]
            patch_ims_path = [os.path.join(data_folder_path, i) for i in patch_ims]
            batch_dict, meta, patch_imgs, input_imgs = preprocess(patch_ims_path, DataVal)
            for k in input_imgs:
                if k == 'batch_size':
                    continue
                input_imgs[k] = torch.from_numpy(input_imgs[k]).to(opt.device)
            output, dets = process(model, input_imgs, return_time, opt, opt.K)
            # 后处理
            rets, dets_post = post_process(dets, meta, num_classes, max_per_image=max_per_image)
            # 后处理
            count = -1
            for ret in rets:
                count=count+1
                scores = ret[1][:,-1]
                ret_filtered = ret[1][scores>conf_filtered,:].reshape(-1,5)
                detections_all[patch_ids[count]] = ret_filtered
        #######tracking and update
        mot_tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.1)
        ids = []
        trajs = []
        for i_im in range(len(img_list)):
            det_coords = detections_all[i_im]
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
            # print(v_mean)
            if v_mean < 0.55:
                continue
            trajs_filt.append(traj_i)

        det_for_images = [[] for i in range(len(img_list))]
        images = [i for i in range(len(img_list))]
        count = 0
        for i_traj in trajs_filt:
            count = count + 1
            for i_trajkk in i_traj:
                index = images.index(i_trajkk[0])
                det_for_images[index].append(i_trajkk[1:] + [count])
        ##################
        for kk in range(len(det_for_images)):
            #
            txt_update_save_name = os.path.join(save_txt_update_folder, img_list[kk].replace('.jpg', '.txt'))
            fid_txt_update = open(txt_update_save_name, 'w')
            # txt_update_save_name_epoch = os.path.join(save_txt_update_folder_epoch, img_list[kk].replace('.jpg', '.txt'))
            # fid_txt_update_epoch = open(txt_update_save_name_epoch, 'w')
            ################################
            #####match for new coords
            coord_ori = detections_ori[kk]
            coord_det = np.array(det_for_images[kk])
            coords_out = match_det_track(coord_ori, coord_det, iou_threshold=0.3)
            ################################
            for i in range(coords_out.shape[0]):
                coord = coords_out[i,:]
                # mask_filt[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])] = 1
                fid_txt_update.write('%d\t%d\t%d\t%d\t0\t%d\n' % (coord[0], coord[1], coord[2], coord[3], -1))
                # fid_txt_update_epoch.write('%d\t%d\t%d\t%d\t0\t%d\n' % (coord[0], coord[1], coord[2], coord[3], -1))
            fid_txt_update.close()
            # fid_txt_update_epoch.close()
        #########
        print(data_folder_list[ii], 'model update done!!!')