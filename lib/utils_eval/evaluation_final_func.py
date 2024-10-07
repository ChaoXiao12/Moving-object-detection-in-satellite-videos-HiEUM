import numpy as np
import scipy.io as sio
import os

import xml.dom.minidom as doxml
from lib.utils1.utils_eval import eval_metric

def eval_func_final(results_dir_tol, data_dir=None, data_name=None, conf_ths = None, write_flag = True, dis_ths = None):
    #eval func
    eval_mode_metric = 'dis'
    if dis_ths is not None:
        dis_th = dis_ths
    else:
        dis_th = [5]
    iou_th = [0.05]
    # conf_thresh_all = [0.2, 0.25, 0.3, 0.32, 0.34, 0.35]
    if conf_ths is not None:
        conf_thresh_all = conf_ths
    else:
        conf_thresh_all = [0.1,0.15,0.2,0.25,0.3]
    if data_name is None:
        dataName = [3,5,2,8,10,6,9]
    else:
        dataName = data_name
    if data_dir is None:
        ANN_PATH0 = '/media/xc/BA61C62ABCE29FF2/xc/dataset/RsCarData/images/test1024/'
    else:
        ANN_PATH0 = data_dir
        data_dir = data_dir.split('images')[0]
    eval_mode = 'fixed'  #'fixed', 'adaptive'
    th_mean = 1
    th_std = 13

    eval_new_mode = 'new'  # 'new' ###选择新的标注进行评测，或者是选择旧的标注进行评测

    conf_results = {}

    for conf_thresh in conf_thresh_all:

        methods_results = {}

        for results_dir0 in results_dir_tol:
            iou_results = []
            print(results_dir0)
            #record the results
            if eval_new_mode == 'new':
                txt_name = 'reuslts_%s_%.2f_F1_new_gt.txt' % (eval_mode_metric, conf_thresh)
            else:
                txt_name = 'reuslts_%s_%.2f_F1.txt' % (eval_mode_metric, conf_thresh)
            if write_flag:
                fid = open(results_dir0 + txt_name, 'w+')
                fid.write(results_dir0 + '(recall,precision,F1)\n')
                fid.write(eval_mode_metric + '\n')
            if eval_mode_metric=='dis':
                thres = dis_th
            elif eval_mode_metric=='iou':
                thres = iou_th
            else:
                raise Exception('Not a valid eval mode!!')
            ##eval
            for thre in thres:#
                thresh_results = {}
                if eval_mode_metric == 'dis':
                    dis_th_cur = thre
                    iou_th_cur = 0.05
                elif eval_mode_metric == 'iou':
                    dis_th_cur = 5
                    iou_th_cur = thre
                else:
                    raise Exception('Not a valid eval mode!!')
                det_metric = eval_metric(dis_th=dis_th_cur, iou_th=iou_th_cur, eval_mode=eval_mode_metric)
                if write_flag:
                    fid.write('conf_thresh=%.2f,thresh=%.2f\n'%(conf_thresh, thre))
                results_temp = {}
                for datafolder in dataName:
                    det_metric.reset()
                    if eval_new_mode == 'new':
                        ANN_PATH = data_dir + 'labeleddata20230227/' + '%03d' % datafolder + '/img1/'
                    else:
                        ANN_PATH = ANN_PATH0 + '%03d' % datafolder + '/xml_det/'
                    # ANN_PATH = ANN_PATH0 + '%03d' % datafolder + '/xml/'
                    if eval_mode == 'adaptive':
                        results_dir = results_dir0 + '%03d/coords_mean_%d_std_%d/' % (datafolder, th_mean, th_std)
                    elif eval_mode == 'fixed':
                        # results_dir = results_dir0 + '%03d/coords/' % (datafolder)
                        results_dir = results_dir0 + '%03d/' % (datafolder)
                    else:
                        raise Exception('Not a valid mode!!!')
                    #start eval
                    anno_dir = os.listdir(ANN_PATH)
                    num_images = len(anno_dir)
                    for index in range(num_images):
                        file_name = anno_dir[index]
                        #load gt
                        if(not file_name.endswith('.xml')):
                            continue
                        annName = ANN_PATH+file_name
                        if not os.path.exists(annName):
                            continue
                        gt_t = det_metric.getGtFromXml(annName)
                        #导入det
                        matname = results_dir + file_name.replace('.xml','.mat')
                        if os.path.exists(matname):
                            det_ori = sio.loadmat(matname)
                            try:
                                det = det_ori['Detect_Result']
                            except:
                                det = det_ori['A']
                            if(det.shape[1]==5):
                                det = np.array(det)
                                score = det[:,-1]
                                inds = np.argsort(-score)
                                score = score[inds]
                                det = det[score>conf_thresh]
                            else:
                                det[:, 2:4] = det[:, 0:2]+det[:, 2:4]
                                det[:, 0:2] = det[:, 0:2]
                        else:
                            det = np.empty([0,4])
                        #更新评价结果
                        det_metric.update(gt_t, det)
                        # print(det_metric.get_result())
                    #获取结果
                    result = det_metric.get_result(img_size=[1024, 1024], seq_len=num_images)
                    if write_flag:
                        fid.write('&%.1f\t&%.1f\t&%.1f\t&%.1f\t&%.2e\t&%.2e\n' % (
                    result['recall'], result['prec'], result['f1'], result['pd'], result['fa_1'], result['fa_2']))
                    print('%s, evalmode=%s, thre=%0.2f, conf_th=%0.2f, re=%0.3f, prec=%0.3f, f1=%0.3f,'
                          ' pd=%0.3f, fa_1=%0.2e, fa_2=%0.2e' % (
                              '%03d' % datafolder, eval_mode_metric, thre, conf_thresh, result['recall'],
                              result['prec'], result['f1'],
                              result['pd'], result['fa_1'], result['fa_2']))
                    results_temp[datafolder] = result
                # 获取 avg results
                meatri = [[v['recall'], v['prec'], v['f1'], v['pd'], v['fa_1'], v['fa_2']] for k, v in
                          results_temp.items()]
                meatri = np.array(meatri)
                avg_results = np.mean(meatri, 0)
                print('avg result:  ', avg_results)
                if write_flag:
                    fid.write(
                    '&%.1f\t&%.1f\t&%.1f\t&%.1f\t&%.2e\t&%.2e\n' % (
                    avg_results[0], avg_results[1], avg_results[2], avg_results[3], avg_results[4], avg_results[5]))
                results_temp['avg'] = {
                    'recall': avg_results[0],
                    'prec': avg_results[1],
                    'f1': avg_results[2],
                    'pd': avg_results[3],
                    'fa1': avg_results[4],
                    'fa2': avg_results[5],
                }
                thresh_results[thre] = results_temp
            methods_results[results_dir0] = thresh_results
        conf_results[conf_thresh] = methods_results
    return conf_results

if __name__ == '__main__':
    results_dir = [
        '/media/xc/DA583A0977A51B46/xc/code/mycode/det/SparseFast/weights/rsdata_multi/sp_centerDet_auto_minus/results/sp_centerDet_auto_minus_Minus_unetv2_decomp__seglen20_weights2022_12_17_10_02_33_model_last_test/',
    ]
    data_dir = '/media/xc/BA61C62ABCE29FF2/xc/dataset/RsCarData/images/test_challenge/'
    data_name =  dataName = [1, 3, 6, 7,  8, 10, 12, 13, 14, 15, 16, 17]
    eval_func_final(results_dir, data_dir, data_name)