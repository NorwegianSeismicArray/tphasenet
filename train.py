# Copyright 2023, Erik Myklebust, Andreas Koehler, MIT license

"""
Code for training phase picking models

"""

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from setup_config import add_root_paths,get_config_dir,dict_to_namespace
from train_utils import CustomStopper
from tensorflow.keras import mixed_precision
from train_utils import get_data_files_new, create_data_generator, get_model, get_predictions

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print('Reading config ...')
config_dir = get_config_dir()
if gpus: config_dir ='tf/'
args = OmegaConf.load(f'{config_dir}/config.yaml')
args_dict = OmegaConf.to_container(args, resolve=True)
args = OmegaConf.create(args_dict)
OmegaConf.set_struct(args, False)
cfg = dict_to_namespace(args)
print('Config read.')

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

GPU = len(tf.config.list_physical_devices('GPU')) > 0

#DATA LOCATION 
if gpus:
    inputdir = 'tf/data/'
else :
    #inputdir = '/projects/Array/ML_methods/waveform_data/'
    inputdir = cfg.data.inputdir

training_years = cfg.data.train_years
testing_years = cfg.data.test_years
validation_years = cfg.data.valid_years

training_files = get_data_files_new(inputdir, training_years, cfg)
validation_files = get_data_files_new(inputdir, validation_years, cfg)
testing_files = get_data_files_new(inputdir, testing_years, cfg )

train_dataset,_ = create_data_generator(training_files, [], cfg, training=True)
valid_dataset,_ = create_data_generator(validation_files, [], cfg, training=True)
test_dataset,nchannels = create_data_generator(testing_files, [], cfg, training=False)

#DEFINING LOGSDIR
from datetime import datetime
datetimestr = datetime.now().strftime('%Y%m%d-%H%M%S')
logs = 'tf/logs/' + datetimestr

model_type = cfg.model.type
dataset = cfg.data.input_dataset_name

#NAME OF EXPERIMENT
setting = f'{dataset}_{model_type}'

#INPUT WAVEFORM SHAPE
input_shape = (int(cfg.data.sampling_rate*cfg.augment.new_size), nchannels)

#GET MODEL AND BUILD WITH INPUT SHAPE
model = get_model(cfg,nchannels)
model.build((cfg.training.batch_size, *input_shape))
print('Num parameters:', model.num_parameters)

monitor_metric = 'val_loss'
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor_metric,
                                                  factor=np.sqrt(0.1),
                                                  min_lr=0.5e-6, 
                                                  mode='min', 
                                                  patience=cfg.training.reduce_lr_patience),
             #EARLY STOPPING WITH WARMUP, i.e., no stopping before 5th epoch.
             CustomStopper(monitor_metric,
                           mode='min',
                           patience=cfg.training.early_stopping_patience, 
                           start_epoch=5,
                           min_delta=1e-4,
                           restore_best_weights=False),
             tf.keras.callbacks.TerminateOnNaN()]

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=cfg.training.epochs,
          callbacks=callbacks
)

model.save(f'tf/outputs/saved_model_{setting}.tf', save_format="tf")

xte, true, pred, meta, metat_mean, metat_std, sample_weight = get_predictions(cfg, test_dataset, model)

ids = [a['event_id'] for a in test_dataset.super_sequence.data]

np.savez(f'tf/outputs/predictions_{setting}.npz', 
         x=xte, 
         y=true, 
         yhat=pred,
         ids=np.array(ids))
