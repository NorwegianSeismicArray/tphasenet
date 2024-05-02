#!/usr/bin/env python
"""
Code for generating input data for ML dataset - public version

"""

print('Importing packages ...')
from tqdm import tqdm
import pandas as pd
from setup_config import add_root_paths,get_config_dir,dict_to_namespace
from omegaconf import OmegaConf
from utils_data import select_arrivals,prepare_windows,get_waveforms
from utils_labels import get_phase_detections
from obspy.clients.fdsn import Client
client = Client('UIB-NORSAR')


if __name__ == '__main__':

    print('Reading config ...')

    config_dir = get_config_dir()
    args = OmegaConf.load(f'{config_dir}/data_config.yaml')
    args_dict = OmegaConf.to_container(args, resolve=True)
    #args_dict = add_root_paths(args_dict)
    args = OmegaConf.create(args_dict)
    OmegaConf.set_struct(args, False)
    cfg = dict_to_namespace(args)
    print('Config read.')

    for year in cfg.select.years:
        print('Loading metadata ...')
        events = pd.read_csv(f'{cfg.run.inputdir}/{cfg.run.input_metadataset_name}_{year}.csv')
        events.rename(columns={'Unnamed: 0': 'event_id'},inplace=True)
        arrivals = pd.read_csv(f'{cfg.run.inputdir}/{cfg.run.input_metadataset_name}_{year}_arrivals.csv')
        arrivals.rename(columns={'Unnamed: 0': 'arrival_id'},inplace=True)

        print('Selecting data ...')
        if cfg.run.verbose : print(f'Number of events/arrivals loaded: {len(events)}/{len(arrivals)}')
        events, arrivals = select_arrivals(events,arrivals,cfg)
        if cfg.run.verbose : print(f'Number of events/arrivals selected: {len(events)}/{len(arrivals)}')

        print("Generating input data ...")
        print('Preparing time windows ...')
        arrivals_in_windows,windows = prepare_windows(events, arrivals,cfg)
        print(f'Getting waveforms for {cfg.data.inputdata_type} ...')
        arrivals_in_windows,windows,nnoise,target_shape = get_waveforms(windows,arrivals_in_windows,year,client,cfg)

        print("Generating label data ...")
        # this requires labels from input data generation above
        get_phase_detections(arrivals_in_windows,windows,target_shape,cfg)

