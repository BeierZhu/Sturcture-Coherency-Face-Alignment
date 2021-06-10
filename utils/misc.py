import os
import shutil
import torch
import math
import numpy as np
from scipy.ndimage import filters
import logging
import cv2
logger = logging.getLogger('FLD')

def save_checkpoint(state, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar', 
                    is_best=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath,
            os.path.join(checkpoint, 'checkpoint_best.pth.tar'))


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters share common prefix 'module.
       If have prefix, then remove"""
    def f(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def add_prefix(state_dict, prefix):
    """If not have prefix, then add"""
    def f(x):
        return prefix + x if not x.startswith(prefix) else x         
    return {f(key): value for key, value in state_dict.items()}


def load_model(ckpt_obj, model, optimizer=None):
    if isinstance(ckpt_obj, dict):
        checkpoint = ckpt_obj
    else:
        logger.info("=> loading checkpoint '{}'".format(ckpt_obj))
        device = torch.cuda.current_device()
        checkpoint = torch.load(ckpt_obj, map_location=lambda storage, loc: storage.cuda(device))

    state_dict = checkpoint['state_dict']
    if not hasattr(model, 'module'):
        state_dict = remove_prefix(state_dict, 'module.')
    else:
        state_dict = add_prefix(state_dict, 'module.')

    model.load_state_dict(state_dict, strict=False)

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        logger.warning('missing keys from checkpoint: {}'.format(k))

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']
        
def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(
        remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    info_str = 'Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (
        i, n, i / n * 100, average_time, remaining_day, remaining_hour, remaining_min)
    logger.info(info_str)

def get_checkpoints(root):
    files = [file for file in os.listdir(root) if file.endswith('.pth.tar') and file.startswith('checkpoint_')]
    files.sort(key=lambda x: int(x[len("checkpoint_"):-len(".pth.tar")]))
    checkpoint_list = [os.path.join(root, file) for file in files]

    return checkpoint_list

def load_mean_pose(file_path):
    with open(file_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                elements = line.strip('\n').strip(' ').split(' ')
                mean_pose = [float(element) for element in elements]
    return np.array(mean_pose)
