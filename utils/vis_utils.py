import os
import logging
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

logger = logging.getLogger('FLD')


def get_logger(logdir):

    logger.info("init logger in {}".format(logdir))
    writer = SummaryWriter(logdir)
    return writer


def get_model_graph(writer, input_shape, net):
    inputs = torch.rand(input_shape).cuda()
    writer.add_graph(net.cuda(), inputs)

 
def add_scalar(writer, scalar_name, val, n_iter):
    writer.add_scalar(scalar_name, val, n_iter)


def add_image(writer, img_name, imgs, n_iter, nrow=1):
    x = vutils.make_grid(imgs, nrow=nrow)
    writer.add_image(img_name, x, n_iter)


def get_show_img(config, input_img, show_idx):
    means = torch.from_numpy(np.array(config.means)).float()
    stds = torch.from_numpy(np.array(config.stds)).float()
    img_norm_factor = float(config.img_norm_factor)
    sample_img = input_img[show_idx].permute(1, 2, 0)
    show_img = ((sample_img * stds + means) * img_norm_factor).permute(2, 0, 1).type(torch.ByteTensor)

    return show_img


def get_hm_tensor(show_idx, input_tensor):
    lst = []
    hm = input_tensor[show_idx]
    for i in range(hm.shape[0]):
        lst.append(hm[i].unsqueeze(0))
    return lst


def save_result_img(image, landmark_pred, landmark_targ, save_path):
    num_kpts = len(landmark_pred) // 2
    for i in range(num_kpts):
        targ_pt = (round(float(landmark_targ[2*i])), round(float(landmark_targ[2*i+1])))
        pred_pt = (round(float(landmark_pred[2*i])), round(float(landmark_pred[2*i+1])))
        # image = cv2.circle(image, targ_pt, 1, (0, 255, 0), 2)
        # image = cv2.circle(image, pred_pt, 1, (0, 0, 255), 2)
        image = cv2.circle(image, pred_pt, 1, (0, 255, 0), 2)

    cv2.imwrite(save_path, image)


def save_result_imgs(images, save_dir, names, landmarks_pred, landmarks_targ):
    for img_id, name in enumerate(names):
        landmark_targ = landmarks_targ[img_id, :]
        landmark_pred = landmarks_pred[img_id, :]
        image = images[img_id,:,:,:]
        result_path = os.path.join(save_dir, name)
        save_result_img(image, landmark_pred, landmark_targ, result_path)


def save_result_nmes(nmes, save_path, names):
    with open(save_path, 'a+') as f:
        for nme, name in zip(nmes, names):
            f.write(str(nme)+' '+name+'\n')


def save_result_lmks(lmks, save_path, names):
    with open(save_path, 'a+') as f:
        for lmk, name in zip(lmks, names):
            for p in lmk:
                f.write(str(p) + ' ')
            f.write(name + '\n')

class CsvHelper(object):
    """docstring for ClassName"""
    def __init__(self, head, save_path):
        super(CsvHelper, self).__init__()
        self.head = head
        self.save_path = save_path
        self.data = []

    def update(self, data):
        self.data.append(data)

    def save(self):
        result = pd.DataFrame(columns=self.head, data=self.data)
        result.to_csv(self.save_path, index=None)
        