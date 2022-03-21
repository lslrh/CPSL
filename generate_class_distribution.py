# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from parser_train import parser_, relative_path_to_absolute_path
import pdb
import math
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from data import create_dataset
from models import adaptation_modelv2
from utils import fliplr

def test(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(opt, logger) 

    if opt.model_name == 'deeplabv2':
        checkpoint = torch.load(opt.resume_path)['ResNet101']["model_state"]
        model = adaptation_modelv2.CustomModel(opt, logger)
        model_dict = {}
        state_dict = model.BaseNet.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.BaseNet.load_state_dict(state_dict)

    validation(model, logger, datasets, device, opt)

def validation(model, logger, datasets, device, opt):
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(datasets.target_train_loader, logger, device, model, opt)
        #validate(datasets.target_valid_loader, device, model, opt)

def label2rgb(func, label):
    rgbs = []
    for k in range(label.shape[0]):
        rgb = func(label[k, 0].cpu().numpy())
        rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1))
    rgbs = torch.stack(rgbs, dim=0).float()
    return rgbs

def validate(valid_loader, logger, device, model, opt):
    ori_LP = os.path.join(opt.save_path)

    if not os.path.exists(ori_LP):
        os.makedirs(ori_LP)

    sm = torch.nn.Softmax(dim=1)
    if opt.class_balance:
        conf_dict, pred_cls_num = val(model.BaseNet_DP, logger, device, opt)
        class_distribution = pred_cls_num/np.sum(pred_cls_num)
        np.save(os.path.join(ori_LP, "class_distribution.npy"), class_distribution)
        #cls_thresh = kc_parameters(conf_dict, pred_cls_num, opt)


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

def val(model, logger, device, opt):
    datasets = create_dataset(opt, logger)
    valid_loader = datasets.target_train_loader
    softmax2d = nn.Softmax2d()
    conf_dict = {k: [] for k in range(opt.n_class)}
    pred_cls_num = np.zeros(opt.n_class)
    sm = nn.Softmax(dim=0)

    for data_i in tqdm(valid_loader):
        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)
        filename = data_i['img_path']
        out = model(images_val)
        
        feat = out['feat']
        bs, _, h, w = feat.shape      
        for i in range(out['out'].shape[0]):
            output_cb = sm(out['out'][i]).cpu().numpy().transpose(1,2,0)
            amax_output = np.asarray(np.argmax(output_cb, axis=2), dtype=np.uint8)
            conf = np.amax(output_cb, axis = 2)
            pred_label = amax_output.copy()
            for idx_cls in range(opt.n_class):
                idx_temp = pred_label == idx_cls
                pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                if idx_temp.any():
                    conf_cls_temp = conf[idx_temp].astype(np.float32)
                    len_cls_temp = conf_cls_temp.size
                    # downsampling by ds_rate
                    conf_cls = conf_cls_temp#[0:len_cls_temp:opt.ds_rate]
                    conf_dict[idx_cls].extend(conf_cls)   
    return conf_dict, pred_cls_num


def kc_parameters(conf_dict, pred_cls_num, opt):
    # threshold for each class
    cls_thresh = np.ones(opt.n_class, dtype = np.float32)
    cls_size = np.zeros(opt.n_class, dtype=np.float32)

    for idx_cls in np.arange(0, opt.n_class):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] != None:
            conf_dict[idx_cls].sort(reverse=True) # sort in descending order
            len_cls = len(conf_dict[idx_cls])
            len_cls_thresh = int(math.floor(len_cls * opt.portion))
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            conf_dict[idx_cls] = None    
    return cls_thresh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--save_path', type=str, default='./Pseudo', help='pseudo label update thred')
    parser.add_argument('--portion', type=float, default= 0.6, help='portion of confident samples')
    parser.add_argument('--soft', action='store_true', help='save soft pseudo label')
    parser.add_argument('--flip', action='store_true')
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')
    opt.noaug = True
    opt.noshuffle = True

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    test(opt, logger)

#python generate_pseudo_label.py --name gta2citylabv2_warmup_soft --soft --resume_path ./logs/gta2citylabv2_warmup/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python generate_pseudo_label.py --name gta2citylabv2_stage1Denoise --flip --resume_path ./logs/gta2citylabv2_stage1Denoisev2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python generate_pseudo_label.py --name gta2citylabv2_stage2 --flip --resume_path ./logs/gta2citylabv2_stage2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast --bn_clr --student_init simclr
#python generate_pseudo_label.py --name syn2citylabv2_warmup_soft --soft --src_dataset synthia --n_class 16 --src_rootpath Dataset/SYNTHIA-RAND-CITYSCAPES --resume_path ./logs/syn2citylabv2_warmup/from_synthia_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
