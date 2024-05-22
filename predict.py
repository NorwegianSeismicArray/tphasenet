# Copyright 2023 Andreas Koehler, Erik Myklebust, MIT license
# Code for making prediction with phase detection models: contineous or window-wise

from obspy.clients.fdsn import Client
import os
import tensorflow as tf
from omegaconf import OmegaConf
from setup_config import get_config_dir,dict_to_namespace
from predict_utils import phase_detection
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file_name',
                    default='predict_config.yaml', help='configuration file')
args = parser.parse_args()
cfg_file = args.config_file_name

client = Client('UIB-NORSAR')

if __name__ == '__main__': 

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print('Reading config ...')
    config_dir = get_config_dir()
    args = OmegaConf.load(f'{config_dir}/{cfg_file}')
    args_dict = OmegaConf.to_container(args, resolve=True)
    args = OmegaConf.create(args_dict)
    OmegaConf.set_struct(args, False)
    cfg_pred = dict_to_namespace(args)
    args = OmegaConf.load(f'{cfg_pred.model}')
    args_dict = OmegaConf.to_container(args, resolve=True)
    args = OmegaConf.create(args_dict)
    OmegaConf.set_struct(args, False)
    cfg_model = dict_to_namespace(args)
    print('Config read.')

    phase_detection(client,cfg_pred,cfg_model)
