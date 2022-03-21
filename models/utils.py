# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from models.sync_batchnorm import SynchronizedBatchNorm2d
import numpy as np

def get_scheduler(optimizer, opt):

    scheduler = PolynomialLR(optimizer, max_iter=opt.train_iters, gamma=0.9)

    return scheduler


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, 
                 gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
        #     return [base_lr for base_lr in self.base_lrs]
        # else:
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
        factor = max(factor, 0)
        return [base_lr * factor for base_lr in self.base_lrs] 

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, scheduler, mode='linear', 
                 warmup_iters=100, gamma=0.2, last_epoch=-1):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == 'linear':
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == 'constant': 
                factor = self.gamma
            else:
                raise KeyError('WarmUp type {} not implemented'.format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs

def cross_entropy2d(input, target, weight=None, size_average=True, softmax_used=False, reduction='mean', cls_num_list=None):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        raise NotImplementedError('sizes of input and label are not consistent')

    # input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # target = target.view(-1)
    if softmax_used:
        loss = F.nll_loss(
            input, target, weight=weight, size_average=size_average, ignore_index=250
        )
    else:
        loss = F.cross_entropy(
            input, target, weight=weight, size_average=size_average, ignore_index=250, reduction=reduction
        )
    return loss

def freeze_bn(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, SynchronizedBatchNorm2d)\
        or isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        
def rand_bbox(mask_shape=(256,512), prop_by_area=True, n_boxes=1, random_aspect_ratio=True,
              within_bounds=True, invert=False):
    prop_range = (0.5, 0.5)
    if prop_by_area:
        # Choose the proportion of each mask that should be above the threshold
        mask_props = np.random.uniform(prop_range[0], prop_range[1])

        # Zeros will cause NaNs, so detect and suppres them
        zero_mask = mask_props == 0.0

        if random_aspect_ratio:
            y_props = np.exp(np.random.uniform(low=0.0, high=1.0) * np.log(mask_props))
            x_props = mask_props / y_props
        else:
            y_props = x_props = np.sqrt(mask_props)
        fac = np.sqrt(1.0 / n_boxes)
        y_props *= fac
        x_props *= fac
    else:
        if random_aspect_ratio:
            y_props = np.random.uniform(prop_range[0], prop_range[1])
            x_props = np.random.uniform(prop_range[0], prop_range[1])
        else:
            x_props = y_props = np.random.uniform(prop_range[0], prop_range[1])
        fac = np.sqrt(1.0 / n_boxes)
        y_props *= fac
        x_props *= fac

    sizes = np.round(np.stack([[y_props], [x_props]], axis=1) * np.array(mask_shape)[None, :])

    if within_bounds:
        positions = np.round((np.array(mask_shape) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(positions, positions + sizes, axis=1)
    else:
        centres = np.round(np.array(mask_shape) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=1)
    return rectangles[0]