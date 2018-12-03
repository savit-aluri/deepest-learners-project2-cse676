
#Library inputs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

import cv2
import numpy as np

# based on arguments provided in input query
def get_arguments():
    args_variable='store_true'
    args_value=['--frequent','--gpus','--workers','--model-file','--use-detect-bbox','--flip-test','--post-process','--shift-heatmap','--coco-bbox-file']
    
    args_help=['frequency of logging','gpus','num of dataloader workers','model state file','use detect bbox','use flip test','use post process','shift heatmap','coco detection bbox file']
    
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True,type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    
    parser.add_argument(args_value[0],help=args_help[0],default=config.PRINT_FREQ,type=int)
    parser.add_argument(args_value[1],help=args_help[1],type=str)
    parser.add_argument(args_value[2],help=args_help[2],type=int)
    parser.add_argument(args_value[3],help=args_help[3],type=str)
    parser.add_argument(args_value[4],help=args_help[4],action=args_variable)
    parser.add_argument(args_value[5],help=args_help[5],action=args_variable)
    parser.add_argument(args_value[6],help=args_help[6],action=args_variable)
    parser.add_argument(args_value[7],help=args_help[7],action=args_variable)
    parser.add_argument(args_value[8],help=args_help[8],type=str) 

    return parser.parse_args()

# Updating configuration for project
def config_sets(config, args):
    values=[args.gpus,args.workers,args.use_detect_bbox,args.flip_test,args.post_process,args.shift_heatmap,args.model_file,args.model_file]
    
    if values[0]:
        config.GPUS = values[0]
    if values[1]:
        config.WORKERS = values[1]
    if values[2]:
        config.TEST.USE_GT_BBOX = not values[2]
    if values[3]:
        config.TEST.FLIP_TEST = values[3]
    if values[4]:
        config.TEST.POST_PROCESS = values[4]
    if values[5]:
        config.TEST.SHIFT_HEATMAP = values[5]
    if values[6]:
        config.TEST.MODEL_FILE = values[6]
    if values[7]:
        config.TEST.COCO_BBOX_FILE = values[7]


def main():
    args = get_arguments()
    config_sets(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')


    # CUDNN setup 
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    model_eval='models.'+config.MODEL.NAME+'.get_pose_net'
    model = eval(model_eval)(config, is_train=False)

    if config.TEST.MODEL_FILE:
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,'final_state.pth.tar')
        model.load_state_dict(torch.load(model_state_file))
    
    # GPU setup for parallel data processing
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # Loss function criterion and optimizer
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading part
    means=[0.485, 0.456, 0.406]
    deviation=[0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=means,std=deviation)
    final_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    final_loader = torch.utils.data.DataLoader(
        final_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(config, final_loader, final_dataset, model, criterion,
             final_output_dir, tb_log_dir)
    
    

if __name__ == '__main__':
    main()
