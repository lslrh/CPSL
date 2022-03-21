# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json

def parser_(parser):
    parser.add_argument('--root', type=str, default='work/Ruihuang', help='root path')
    parser.add_argument('--model_name', type=str, default='deeplabv2', help='deeplabv2')
    parser.add_argument('--name', type=str, default='gta2city', help='pretrain source model')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--train_iters', type=int, default=90000)
    parser.add_argument('--moving_prototype', action='store_true')
    parser.add_argument('--bn', type=str, default='sync_bn', help='sync_bn|bn|gn|adabn')
    #training
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--seed', type=int, default=6523, help='random seed')
    parser.add_argument('--stage', type=str, default='stage1', help='warm_up|stage1|stage2|stage3')
    parser.add_argument('--finetune', action='store_true')
    #model
    parser.add_argument('--resume_path', type=str, default='pretrained/from_gta5_to_cityscapes_on_deeplab101_best_model_warmup.pkl', help='resume model path')
    parser.add_argument('--ema', action='store_true', help='use ema model')
    parser.add_argument('--ema_bn', action='store_true', help='add extra bn for ema model')
    parser.add_argument("--student_init", default='stage1', type=str, help="stage1|imagenet|simclr")
    parser.add_argument("--proto_momentum", default=0.0001, type=float)
    parser.add_argument("--bn_clr", action='store_true', help="if true, add a bn layer for the output of simclr model")
    #data
    parser.add_argument('--src_dataset', type=str, default='gta5', help='gta5|synthia')
    parser.add_argument('--tgt_dataset', type=str, default='cityscapes', help='cityscapes')
    parser.add_argument('--src_rootpath', type=str, default='./dataset/GTA5')
    parser.add_argument('--tgt_rootpath', type=str, default='./dataset/cityscapes')
    parser.add_argument('--path_LP', type=str, default='./Pseudo/gta2citylabv2_warmup_soft', help='path of probability-based PLA')
    parser.add_argument('--path_soft', type=str, default='./Pseudo/gta2citylabv2_warmup_soft', help='soft pseudo label for rectification')
    parser.add_argument("--train_thred", default=0, type=float)
    parser.add_argument('--used_save_pseudo', action='store_true', help='if True used saved pseudo label')
    parser.add_argument('--no_droplast', action='store_true')

    parser.add_argument('--resize', type=int, default=2200, help='resize long size')
    parser.add_argument('--rcrop', type=str, default='896,512', help='rondom crop size')
    parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')
    parser.add_argument('--jitter', type=float, default=0.4, help='random jitter')

    parser.add_argument('--n_class', type=int, default=19, help='19|16|13')
    parser.add_argument('--num_workers', type=int, default=6)
    #loss
    parser.add_argument('--gan', type=str, default='LS', help='Vanilla|LS')
    parser.add_argument('--adv', type=float, default=0.01, help='loss weight of adv loss, only use when stage=warm_up')
    parser.add_argument('--S_pseudo_src', type=float, default=0.0, help='loss weight of pseudo label for strong augmentation of source')
    parser.add_argument("--rce", action='store_true', help="if true, use symmetry cross entropy loss")
    parser.add_argument("--rce_alpha", default=0.1, type=float, help="loss weight for symmetry cross entropy loss")
    parser.add_argument("--rce_beta", default=1.0, type=float, help="loss weight for symmetry cross entropy loss")
    parser.add_argument("--regular_w", default=0, type=float, help='loss weight for regular term')
    parser.add_argument("--regular_type", default='MRKLD', type=str, help='MRENT|MRKLD')
    parser.add_argument('--proto_consistW', type=float, default=1.0, help='loss weight for proto_consist')
    parser.add_argument("--distillation", default=0, type=float, help="kl loss weight")
    parser.add_argument("--SL_lambda", default=0.1, type=float, help="parameter for self labeling")
    parser.add_argument('--S_pseudo', type=float, default=0.0, help='loss weight of pseudo label for strong augmentation')
    parser.add_argument("--contrast_lambda", default=0.001, type=float, help="parameter for contrastive loss")
    
    # self label
    parser.add_argument("--fbs", default=256, type=int, help="feature batch size")
    parser.add_argument("--queue_length", default=65536, type=int, help="length of queue")
    parser.add_argument("--ds_rate", default=4, type=int, help="the rate of downsampling")
    parser.add_argument("--temperature", default=0.08, type=float, help="the parameter to scale logit")
    parser.add_argument("--epsilon", default=0.05, type=float, help="the parameter to scale logit")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int, help="sinkhorn iterations")
    
    #print
    parser.add_argument('--print_interval', type=int, default=20, help='print loss')
    parser.add_argument('--val_interval', type=int, default=1000, help='validate model iter')
    parser.add_argument('--update_interval', type=int, default=1, help='update model iter')

    parser.add_argument('--noshuffle', action='store_true', help='do not use shuffle')
    parser.add_argument('--noaug', action='store_true', help='do not use data augmentation')

    parser.add_argument('--proto_rectify', action='store_true')
    parser.add_argument('--class_balance', action='store_true')
    parser.add_argument('--proto_temperature', type=float, default=1.0)
    #stage2
    parser.add_argument("--threshold", default=0, type=float)
    return parser

def relative_path_to_absolute_path(opt):
    opt.rcrop = [int(opt.rcrop.split(',')[0]), int(opt.rcrop.split(',')[1])]
    opt.resume_path = os.path.join(opt.resume_path)
    opt.src_rootpath = os.path.join(opt.src_rootpath)
    opt.tgt_rootpath = os.path.join(opt.tgt_rootpath)
    opt.path_LP = os.path.join(opt.path_LP)
    opt.path_soft = os.path.join(opt.path_soft)
    opt.logdir = os.path.join('logs', opt.name)
    return opt