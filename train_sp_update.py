from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data
from lib.utils1.opts import opts
from lib.utils1.logger import Logger
from datetime import datetime

from lib.models.stNet import get_det_net,load_model, save_model
from lib.dataset.dataset_factory import get_dataset
from lib.Trainer.trainer_factory import get_trainer

from lib.LRSD.filter_lrsd import generate_labels

def main(opt):
    torch.manual_seed(opt.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    ###
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    opt.save_dir = opt.save_dir + '/' + opt.datasetname+'_'+opt.data_mode

    if (not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    opt.save_dir = opt.save_dir + '/' + opt.model_name

    if (not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    opt.save_results_dir = opt.save_dir + '/results'

    model_path_name = opt.exp_name +'_supMode_%d' % opt.sup_mode + '_seglen%d_' % opt.seqLen + 'weights' + time_str

    opt.save_dir = opt.save_dir + '/'+model_path_name
    opt.save_log_dir = opt.save_dir


    ###

    val_intervals = opt.val_intervals

    dataset = get_dataset(opt)


    DataVal = dataset(opt, 'test')

    val_loader = torch.utils.data.DataLoader(
        DataVal,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    DataTrain = dataset(opt, 'train')

    base_s = DataTrain.coco

    train_loader = torch.utils.data.DataLoader(
        DataTrain,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Creating model...')
    if opt.off_flag:
        head = {'hm': DataTrain.num_classes, 'wh': 2, 'reg': 2}
    else:
        head = {'hm': DataTrain.num_classes, 'wh': 2}
    model = get_det_net(head, opt.model_name, DataTrain.resolution, opt.seqLen, opt)  # 建立模型

    print(head)
    print(opt.model_name)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)  #设置优化器

    start_epoch = 0

    if(not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    if(not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    logger = Logger(opt)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)  # 导入训练好的模型

    trainer = get_trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    print('Starting training...')

    best = -1

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        if opt.sup_mode==3 and epoch == 1:
            print('generate labels!!!')
            generate_labels(opt.data_dir)
            print('generate labels done!!!')

        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('epoch: {} |'.format(epoch))


        model_save_path = os.path.join(opt.save_dir, 'model_last.pth')
        save_model(model_save_path,
                   epoch, model, optimizer)

        if epoch%opt.unsup_iter==0 and opt.sup_mode==3:
            trainer.update_label(epoch, val_loader, base_s, DataVal, model_save_path)

        for k, v in log_dict_train.items():
            logger.write('{} {:8f} | '.format(k, v))
        if val_intervals > 0 and epoch % val_intervals == 0:

            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds, stats = trainer.val(epoch, val_loader, base_s, DataVal, model_save_path, model_path_name)
            logger.write('eval results: ')
            for k, v in log_dict_val.items():
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val['f1_best'] > best:
                best = log_dict_val['f1_best']
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        logger.write('\n')
        if epoch in opt.lr_step:
            # save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
            #            epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)