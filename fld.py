import os
import cv2
import time
import shutil
import logging
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import WeightedRandomSampler

from models.build_model import ModelBuilder
from datasets.fld_dataset import FLDDS, TransformBuilder
from losses.wing_loss import WingLoss, SmoothWingLoss, WiderWingLoss, NormalizedWiderWingLoss, L2Loss, EuclideanLoss, NMELoss, LaplacianLoss
from utils.log_helper import init_log
from utils.vis_utils import save_result_imgs, save_result_nmes, save_result_lmks
from utils.vis_utils import get_logger, add_scalar, get_model_graph, CsvHelper
from utils.misc import save_checkpoint, print_speed, load_model, get_checkpoints
from utils.metrics import MiscMeter, eval_NME
from utils.imutils import refine_300W_landmarks

init_log('FLD')
logger = logging.getLogger('FLD')

class FLD(object):
    def __init__(self, task_config):
        self.config = EasyDict(task_config)
        cudnn.benchmark = True
        self._build()

    def train(self):
        config = self.config.train_param
        num_kpts = config.num_kpts
        lr_scheduler = self.lr_scheduler
        train_loader = self.train_loader
        model = self.model
        model.train()
        ION_MIN = MiscMeter()
        IPN_MIN = MiscMeter()

        for epoch in range(self.start_epoch, config.scheduler.epochs):
            lr_scheduler.step()
            lr = lr_scheduler.get_lr()[0]
            # train for one epoch
            batch_time = MiscMeter()
            data_time = MiscMeter()
            task_losses = MiscMeter()
            laplace_losses = MiscMeter()
            ION = MiscMeter()            
            IPN = MiscMeter()
            end = time.time()
            for i, samples in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                # retrive data 
                image = samples['image']
                landmarks = samples['landmarks']
                # forward
                landmarks_pred = model(image.cuda())
                # compute loss
                task_loss = self.task_loss(landmarks_pred, landmarks.cuda()) * config.task_weight
                laplace_loss = self.laplace_loss(landmarks_pred, landmarks.cuda()) * config.laplace_weight

                loss = task_loss + laplace_loss
                task_losses.update(task_loss.item())
                laplace_losses.update(laplace_loss.item())
                # compute gradient and do SGD step
                lr_scheduler.optimizer.zero_grad()
                loss.backward()
                lr_scheduler.optimizer.step()
                # compute statistics
                landmarks_pred = landmarks_pred.cpu().data
                ion, _ = eval_NME(landmarks_pred.numpy(), landmarks.numpy(), num_kpts, mode='IO')
                ipn, _ = eval_NME(landmarks_pred.numpy(), landmarks.numpy(), num_kpts, mode='IP')
                ION.update(ion)
                IPN.update(ipn)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                data_time_ratio = data_time.avg/batch_time.avg
                # print info
                if i % config.print_freq == 0:
                    logger.info('Epoch: [{0}][{1}/{2}] '
                                'Data Time Ratio: {data_time_ratio:.3f} '
                                'ION {ION.avg:.4f} '
                                'IPN {IPN.avg:.4f} '
                                'LR {lr:.7f} '
                                'Task Loss {task_loss.avg:.4f} '
                                'Laplace Loss {laplace_loss.avg: 4f}'.format(epoch, i, len(train_loader),
                                                            data_time_ratio=data_time_ratio,
                                                            ION=ION,
                                                            IPN=IPN,
                                                            task_loss=task_losses,
                                                            laplace_loss=laplace_losses,
                                                            lr=lr))
                    print_speed(
                        epoch * len(train_loader) + i + 1,
                        batch_time.val,
                        config.scheduler.epochs * len(train_loader))

            if (epoch + 1) % config.snapshot_freq == 0:
                # deallocate memory
                del loss, landmarks_pred
                # evaluate running models
                ion, ipn = self.evaluate()
                is_best = True if ion < ION_MIN.min else False
                # save current checkpoint and best checkpoint
                ION_MIN.update(ion)
                IPN_MIN.update(ipn)
                logger.info('[TEST] ION min {:.4f} IPN min {:.4f}'.format(ION_MIN.min, IPN_MIN.min))
                model.train()

                save_checkpoint({
                        'epoch': epoch + 1,
                        'optimizer': lr_scheduler.optimizer.state_dict(),
                        'state_dict': model.state_dict(),}, 
                        checkpoint=self.expname, is_best=is_best)
            model.cuda()

    @torch.no_grad()
    def evaluate(self):
        config = self.config
        if config.visualize:
            self._vis_dir_setting()
        self._report_path_setting()
        num_kpts = config.structure.num_kpts
        model = self.model
        model.eval()
        data_loader = self.val_loader
        ION = MiscMeter() 
        IPN = MiscMeter() 

        for batch_id, samples in enumerate(data_loader):
            image = samples['image']
            landmarks_targ = samples['landmarks'].numpy()
            names = samples['name']
            batch_size = image.shape[0]
            landmarks_pred = model(image.cuda())
            landmarks_pred = landmarks_pred.cpu().data.numpy()
            if hasattr(config.test_param, 'refine_300W') and config.test_param.refine_300W:
                landmarks_pred = refine_300W_landmarks(samples['image_cv2'].numpy(), landmarks_pred)
            ion, ions = eval_NME(landmarks_pred, landmarks_targ, num_kpts, mode='IO')
            ipn, ipns = eval_NME(landmarks_pred, landmarks_targ, num_kpts, mode='IP')
            ION.update(ion, batch_size)
            IPN.update(ipn, batch_size)
            save_result_nmes(ions, self.nme_path, names)
            save_result_lmks(landmarks_targ, self.lmk_path, names)
            if config.visualize:
                images_cv2 = samples['image_cv2'].numpy()
                save_result_imgs(images_cv2, self.vis_dir, names, landmarks_pred, landmarks_targ)

        logger.info('ION: {ION.avg:.4f} IPN: {IPN.avg:.4f} '.format(ION=ION, IPN=IPN))
        return ION.avg, IPN.avg

    @torch.no_grad()
    def eval_ckpts(self):
        expname = self.config.expname
        ION = MiscMeter()
        IPN = MiscMeter()
        csv_helper = CsvHelper(head=['Name', 'ION', 'IPN'], 
                               save_path=os.path.join(expname, 'ckpts.csv'))
        for model_path in get_checkpoints(expname):
            load_model(model_path, self.model)
            ion, ipn = self.evaluate()
            ION.update(ion)
            IPN.update(ipn)
            csv_helper.update([model_path, ion, ipn]) 
        csv_helper.save()
        logger.info('Min ION: {ION.min:.4f} '
                    'Min IPN: {IPN.min:.4f} '.format(ION=ION, IPN=IPN))
    
    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()
        if config.train:
            self._build_optimizer()
            self._load_model()
            self._build_scheduler()
            self._build_criterion()
            self._build_tensorboard()
        else:
            self._load_model()
        self._build_dataloader()

    def _load_model(self):
        config = self.config
        if hasattr(config, 'load_path') and config.load_path:
            if os.path.isfile(config.load_path):
                if hasattr(config, 'resume') and config.resume:
                    self.start_epoch = load_model(config.load_path, self.model, self.optimizer)
                else:
                    load_model(config.load_path, self.model)
            else:
                logger.error("=> no checkpoint found at '{}'".format(config.load_path))
                exit(-1)

    def _build_model(self):
        config = self.config
        self.model = FLD.build_model_helper(config=config)

    def _build_criterion(self):
        config = self.config.train_param.criterion
        num_points = self.config.train_param.num_kpts
        if config.type == 'SmoothL1':
            self.task_loss = nn.SmoothL1Loss().cuda()
        elif config.type == 'MSE':
            self.task_loss = nn.MSELoss().cuda()  
        elif config.type == 'NME':
            self.task_loss = NMELoss(**config.kwargs).cuda()  
        elif config.type == 'L1':
            self.task_loss = nn.L1Loss().cuda()
        elif config.type == 'L2':    
            self.task_loss = L2Loss().cuda()
        elif config.type == 'Euclidean':    
            self.task_loss = EuclideanLoss().cuda()
        elif config.type == 'Wing':
            self.task_loss = WingLoss(**config.kwargs).cuda()
        elif config.type == 'SmoothWing':
            self.task_loss = SmoothWingLoss(**config.kwargs).cuda()
        elif config.type == 'WiderWing':
            self.task_loss = WiderWingLoss(**config.kwargs).cuda()
        elif config.type == 'NormalizedWiderWing':    
            self.task_loss = NormalizedWiderWingLoss(**config.kwargs).cuda()
        
        self.laplace_loss = LaplacianLoss(WiderWingLoss(**config.kwargs).cuda(), num_points=num_points).cuda()

    @staticmethod
    def build_model_helper(config=None):
        model = ModelBuilder(config.structure)
        if torch.cuda.device_count() > 1:
            logger.debug("using {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model = model.cuda()
        return model

    def _build_optimizer(self):
        config = self.config.train_param.optimizer
        model = self.model
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)
        optimizer = optim(model.parameters(), **config.kwargs)
        self.optimizer = optimizer

    def _build_train_loader(self):
        config = self.config.train_param
        train_transforms = self.transform_builder.train_transforms()
        is_pdb = True if hasattr(config, 'is_pdb') else False
        num_bins = config.is_pdb if hasattr(config, 'is_pdb') else 9
        train_dataset = FLDDS(
            root_folders=config.train_root,
            sources=config.train_source,
            transform=train_transforms,
            num_points=config.num_kpts,
            is_pdb=is_pdb, num_bins=num_bins)

        self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=False, drop_last=True)

    def _build_val_loader(self):
        config = self.config.test_param
        test_transforms = self.transform_builder.test_transforms()
        test_dataset = FLDDS(
            root_folders=config.val_root,
            sources=config.val_source,
            transform=test_transforms,
            num_points=config.num_kpts)

        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=False)

    def _build_dataloader(self):
        config = self.config
        self.transform_builder = TransformBuilder(config)
        if config.train:
            self._build_train_loader()
            self._build_val_loader()
        else:
            self._build_val_loader()

    def _build_scheduler(self):
        config = self.config.train_param
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.gamma,
            last_epoch=self.start_epoch - 1)

    def _dir_setting(self):
        config = self.config
        self.expname = config.expname
        if not os.path.isdir(self.expname):
            os.mkdir(self.expname)

    def _vis_dir_setting(self):
        '''Set up the directory for saving landmarked images'''
        self.vis_dir = os.path.join(self.expname, 'vis_result')
        if not os.path.isdir(self.vis_dir):
            os.mkdir(self.vis_dir)

    def _report_path_setting(self):
        self.nme_path = os.path.join(self.expname, 'nme.txt')
        self.lmk_path = os.path.join(self.expname, 'lmk.txt')
        if os.path.exists(self.nme_path):
            os.remove(self.nme_path)
        if os.path.exists(self.lmk_path):
            os.remove(self.lmk_path)

    def _build_tensorboard(self):
        logger_path = os.path.join(self.expname, 'log/')
        if os.path.exists(logger_path):
            shutil.rmtree(logger_path)
        os.mkdir(logger_path)
        self.tb_logger = get_logger(logger_path)


