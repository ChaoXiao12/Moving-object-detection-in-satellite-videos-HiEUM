from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.dataset_factory import get_dataset
from lib.utils_eval.evaluation_final_func import eval_func_final
from lib.test_utils.show_imgs import *
from lib.test_utils.process_img_dets import *

import GPUtil
import scipy.io as scio

def test(opt, split, modelPath, show_flag, results_name, save_mat=False, i_th=3):

    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    opt.test_large_size = True

    print(opt.model_name)

    dataset = get_dataset(opt)

    DataVal = dataset(opt, split)
    if opt.off_flag:
        head = {'hm': DataVal.num_classes, 'wh': 2, 'reg': 2}
    else:
        head = {'hm': DataVal.num_classes, 'wh': 2}
    model = get_det_net(head, opt.model_name, DataVal.resolution, opt.seqLen, opt, thresh=i_th)  # 建立模型
    model = load_model(model, modelPath)
    model = model.to(opt.device)
    model.eval()


    return_time = False
    num_classes = dataset.num_classes
    max_per_image = opt.K

    if save_mat:
        save_mat_path_upper = os.path.join(opt.save_results_dir, results_name)
        if not os.path.exists(save_mat_path_upper):
            os.mkdir(save_mat_path_upper)

    test_upper_path = opt.data_dir + 'images/test1024/'

    data_folder_list = os.listdir(test_upper_path)
    patch_len = opt.seqLen

    time_all = []

    for ii in range(len(data_folder_list)):
        data_folder_path = os.path.join(test_upper_path, data_folder_list[ii], 'img1')
        if save_mat:
            save_mat_folder = os.path.join(save_mat_path_upper, data_folder_list[ii])
            if not os.path.exists(save_mat_folder):
                os.mkdir(save_mat_folder)
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
        for pk in range(patch_num):
            time_start = time.time()
            if overlap_flag and pk==patch_num-1:
                patch_ims = img_list[imgs_number-patch_len:imgs_number]
            else:
                patch_ims = img_list[pk*patch_len : (pk+1)*patch_len]
            patch_ims_path = [os.path.join(data_folder_path, i) for i in patch_ims]
            batch_dict, meta, patch_imgs, input_imgs = preprocess(patch_ims_path, DataVal)
            for k in input_imgs:
                if k == 'batch_size':
                    continue
                input_imgs[k] = torch.from_numpy(input_imgs[k]).to(opt.device)
            time_start1 = time.time()
            output, dets = process(model, input_imgs, return_time, opt, opt.K)
            torch.cuda.synchronize()
            time_start3 = time.time()
            # 后处理
            rets, dets_post = post_process(dets, meta, num_classes, max_per_image=max_per_image)
            time_end = time.time()
            print('time_used:', time_end - time_start, time_end - time_start1, time_start3 - time_start1)
            time_all.append(time_end - time_start1)
            gpus = GPUtil.getGPUs()
            gpu = gpus[0]
            print('patch_len: {} GPU used: {}/{}'.format(patch_len, gpu.memoryUsed, gpu.memoryTotal))
            ### view results
            if save_mat:
                fig_save_name1 = os.path.join(save_mat_folder, '%03d_ori.png'%(pk+1))
                fig_save_name2 = os.path.join(save_mat_folder, '%03d_det.png' % (pk + 1))
                view_cloud(output['voxel_coords'], save_flag=1, fig_save_name = fig_save_name1)
                view_dets(dets, conf_th=0.3,save_flag=1, fig_save_name = fig_save_name2)
            if(show_flag):
                hm1 = output['hm'].squeeze(0).squeeze(0).cpu().detach().numpy()
                for det_i in range(len(dets_post)):
                    img = patch_imgs[:,:,:,det_i]
                    frame, _ = cv2_demo(img.astype(np.uint8), dets_post[det_i][1])

                    cv2.imshow('frame',frame)
                    cv2.waitKey(5)
                    hm2 = hm1[det_i]
                    cv2.imshow('hm', hm2)
                    cv2.waitKey(5)

            if save_mat:
                for ik in range(len(patch_ims)):
                    mat_save_name = os.path.join(save_mat_folder, patch_ims[ik].replace('.jpg', '.mat'))
                    ret = rets[ik]
                    A = np.array(ret[1])
                    scio.savemat(mat_save_name, {'A':A})

    time_mean = np.array(time_all).mean()
    print('total_time_mean:', time_mean/patch_len, 'frames per second: ', 1/time_mean*patch_len)

    conf_results = eval_func_final([os.path.join(opt.save_results_dir, results_name + '/')], data_dir=test_upper_path)
    results_return = {}
    best = -1
    for conf, v_c in conf_results.items():
        for k_m, v_m in v_c.items():
            for k_d, v_d in v_m.items():
                re = v_d['avg']['recall']
                pre = v_d['avg']['prec']
                f1 = v_d['avg']['f1']
                results_return['conf_%.2f'%conf + '_avg_recall'] = re
                results_return['conf_%.2f' % conf + '_avg_prec'] = pre
                results_return['conf_%.2f' % conf + '_avg_f1'] = f1
                if best < f1:
                    best = f1
    results_return['f1_best'] = best
    results_return['total_time_mean'] = time_mean/patch_len
    results_return['frames_per_second'] = 1/time_mean*patch_len

    if save_mat:
        results_tol_txt = os.path.join(opt.save_results_dir, results_name, 'results_tol.txt')
        results_tol_txt_fid = open(results_tol_txt, 'w+')
        results_tol_txt_fid.write(results_name+'\n')
        for k,v in results_return.items():
            results_tol_txt_fid.write(k+': %.4f\n'%v)


        results_tol_txt_fid.close()

        time_txt = open(os.path.join(opt.save_results_dir, results_name, 'time.txt'),'w')
        time_txt.write('total_time_mean: %.4f\t frames per second: %.2f\n'%(time_mean/patch_len, 1/time_mean*patch_len))
        time_txt.close()

    return results_return