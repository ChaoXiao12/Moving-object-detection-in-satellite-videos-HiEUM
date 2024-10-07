from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils1.opts import opts
import torch
import os

from lib.test_utils.test import test
from lib.test_utils.test_update import test_update

if __name__ == '__main__':

    opt = opts().parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    split = 'test'
    show_flag = opt.show_results
    savemat = True

    opt.save_dir = opt.save_dir + '/' + opt.datasetname+'_'+opt.data_mode
    if (not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)
    opt.save_dir = opt.save_dir + '/' + opt.model_name
    if (not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)
    opt.save_results_dir = opt.save_dir + '/results'

    if (not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    modelPath = opt.load_model

    print(modelPath)

    results_name = opt.model_name + '_' + modelPath.split('/')[-2] + \
                   '_' + modelPath.split('/')[-1].split('.')[0]

    # test_update(opt, split, modelPath, show_flag, results_name, save_mat=False, epoch=0)

    results_return = test(opt, split, modelPath, show_flag, results_name, savemat)
