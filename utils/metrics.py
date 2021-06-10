import numpy as np
import torch

class MiscMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 9e15
        self.max = -9e15

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = val if val < self.min else self.min
        self.max = val if val > self.max else self.max


def eval_NME(pred, targ, num_points=68, mode='IO'):
    """Evaluation metric for face alignment, numpy version
    Args:
        pred (numpy - shape of batchsize x num_points*2 x 1): predicted face landmarks 
        targ (numpy - shape of batchsize x num_points*2 x 1): target face landmarks
        num_points (int): # face landmarks
        mode (string): 'IP' inter pupil 
                       'IO' inter ocular
    Return:
        (float): normalized mean error according to mode
        (numpy array):
        (int):
    """
    assert num_points in [29, 68, 98, 106]
    if num_points == 29 and mode == 'IO':
        le_idx = [8]
        re_idx = [9]

    if num_points == 29 and mode == 'IP':
        le_idx = [16]
        re_idx = [17]
        
    if num_points == 68 and mode == 'IO':
        le_idx = [36] # le = left eye
        re_idx = [45] # re = right eye

    if num_points == 68 and mode == 'IP':
        le_idx = [36, 37, 38, 39, 40, 41]
        re_idx = [42, 43, 44, 45, 46, 47]

    if num_points == 98 and mode == 'IO':
        le_idx = [60]
        re_idx = [72]

    if num_points == 98 and mode == 'IP':
        le_idx = [96]
        re_idx = [97]

    if num_points == 106: # NME for 106 pts cal the distance between eyes and mouth 
        le_idx = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
        re_idx = [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                  96, 97, 98, 99, 100, 101, 102, 103]

    pred_vec2 = pred.reshape(-1, num_points, 2) # batchsize x 68 x 2
    targ_vec2 = targ.reshape(-1, num_points, 2)

    le_loc = np.mean(targ_vec2[:,le_idx,:], axis=1) # batchsize x 2
    re_loc = np.mean(targ_vec2[:,re_idx,:], axis=1)  

    norm_dist = np.sqrt(np.sum((le_loc - re_loc)**2, axis=1))  # batchsize
    mse = np.mean(np.sqrt(np.sum((pred_vec2 - targ_vec2)**2, axis=2)), axis=1) # batchsize

    nme = mse / norm_dist

    return np.mean(nme), nme