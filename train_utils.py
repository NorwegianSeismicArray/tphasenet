# Copyright 2023, Erik Myklebust, Andreas Koehler, MIT license

import tensorflow as tf
import h5py
from tqdm import tqdm
import numpy as np
from random import shuffle, choices
from collections import defaultdict
import random
import string
import models as nm
from omegaconf.errors import ConfigAttributeError
from scipy.signal.windows import tukey, gaussian, triang
import os
import glob

def keras_f1(y_true, y_pred):
    
    """ 
    
    Calculate F1-score.
    
    Parameters
    ----------
    y_true : 1D array
        Ground truth labels. 
        
    y_pred : 1D array
        Predicted labels.     
        
    Returns
    -------  
    f1 : float
        Calculated F1-score. 
        
    """     
    
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-1)
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def waveform_normalize(X, mode='max', channel_mode='local'):
    """ 
    
    Normalizes waveform data.
    
    Parameters
    ----------
    X : numpy array
        waveforms
        
    mod : string
          mode of normalization : 'max' or 'std'

    channel_mode : string
         'local' or 'global' normalization
        
    Returns
    -------  
    X / m : numpy array
        Normalized waveforms
        
    """

    X -= np.mean(X, axis=0, keepdims=True)

    if mode == 'max':
        if channel_mode == 'local':
            m = np.max(X, axis=0, keepdims=True)
        else:
            m = np.max(X, keepdims=True)
    elif mode == 'std':
        if channel_mode == 'local':
            m = np.std(X, axis=0, keepdims=True)
        else:
            m = np.std(X, keepdims=True)
    else:
        raise NotImplementedError(
            f'Not supported normalization mode: {mode}')

    m[m == 0] = 1
    return X / m

def waveform_crop(x, y, new_length, testing=False):
    """ 
    
    crops waveform data.
    
    Parameters
    ----------
    x : numpy array
        waveforms
    y : numpy array
        labels
        
    new_length : int
          number of samples to be cropped to 

    Returns
    -------  
    x,y : numpy arrays
        cropped waveforms and labels
        
    """

    if testing: 
        y1 = len(x)//2 - new_length//2 #consistent for testing
    else:
        y1 = np.random.randint(0, len(x) - new_length)
    x = x[y1:y1 + new_length]
    y = y[y1:y1 + new_length]
    return x, y

def waveform_drop_channel(x, channel):
    """ 
    
    drops channel from waveform data.
    
    Parameters
    ----------
    x : numpy array
        waveforms
        
    channel : int
          channel number to be dropped 

    Returns
    -------  
    x : numpy array
        waveforms with one chnnel set to zero
        
    """

    x[..., channel] = 0
    return x

def waveform_add_gap(x, max_size):
    """ 
    
    add gap in waveform data.
    
    Parameters
    ----------
    x : numpy array
        waveforms
        
    max_size : int
          length of maximum gap to be added

    Returns
    -------  
    x : numpy array
        waveforms with gap
        
    """

    l = x.shape[0]
    gap_start = np.random.randint(0, int((1 - max_size) * l))
    gap_end = np.random.randint(gap_start, gap_start + int(max_size * l))
    x[gap_start:gap_end] = 0
    return x

def waveform_add_noise(x, noise):
    """ 
    
    adds noise to waveform data.
    
    Parameters
    ----------
    x : numpy array
        waveforms
        
    noise : float
          noise factor 

    Returns
    -------  
    x : numpy array
        waveforms with noise added
        
    """

    m = x.max(axis=0)
    N = np.random.normal(scale=m * noise, size=x.shape)
    return x + N

def waveform_taper(x, alpha=0.04):
    """
    taper waveforms

    """
    w = tukey(x.shape[0], alpha)
    return x*w[:,np.newaxis]

def label_smoothing(y, f):
    """

    smooth phase detections labels by convolution with gaussian
    this adds uncertainty to picked times
    
    Parameters
    ----------
    y : numpy array
        labels
        
    f : numpy array
          gaussian controlled by ramp parameter in config file

    Returns
    -------  
    y : numpy array
        smoothed labels

    """
    y = np.asarray([np.convolve(b, f, mode='same') for b in y.T]).T
    m = np.amax(y, axis=0, keepdims=True)
    m[m == 0] = 1
    y /= m
    return np.clip(y, 0.0, 1.0)

class EQDatareader(tf.keras.utils.Sequence):
    """

    Class for loading data for model training in Keras

    """
    def __init__(self, 
                 files, 
                 augment=False,
                 batch_size=32,
                 new_length=None,
                 taper_alpha=0.01,
                 add_noise=0.0, 
                 add_event=0.0,
                 drop_channel=0.0,
                 add_gap=0.0,
                 max_gap_size=0.0,
                 norm_mode='max',
                 norm_channel_mode='global',
                 fill_value=0.0,
                 include_noise=False,
                 testing=False,
                 ramp=0,
                 file_buffer=-1,
                 shuffle=False) -> None:

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.augment = augment
        self.new_length = new_length
        self.taper_alpha = taper_alpha
        self.add_noise = add_noise
        self.drop_channel = drop_channel
        self.add_event = add_event
        self.add_gap = add_gap
        self.norm_mode = norm_mode
        self.norm_channel_mode = norm_channel_mode
        self.fill_value = fill_value
        self.max_gap_size = max_gap_size
        self.include_noise = include_noise
        self.testing = testing 
        self.ramp = ramp 
        self.f = gaussian(201, ramp)
        self.file_buffer = file_buffer
        self.files = files
        
        self.data = []

        if self.file_buffer < 0:
            for file in tqdm(self.files,total=len(self.files)):
                    self._load_file(file)
        print('Number of samples:', len(self.data))
        self.on_epoch_end()

    def _load_file(self, filename):
        with h5py.File(filename) as f:
            x = f['X'][:]
            ids = [None]*len(x)
            event_type = f['event_type'][:]
            labels = f['label'][:]

        keep_dummy_data_for_github = False
        if keep_dummy_data_for_github :
          with h5py.File(filename.split('/')[-1], 'w') as f:
            f.create_dataset('X', data=x[:1], dtype='float32')
            f.create_dataset('event_type', data=event_type[:1])
            f.create_dataset('label', data=labels[:1])

        if len(x) != len(labels) :
            print(f"Data and labels are not equal {len(ids)} {len(ids2)}")
            exit()
        for _id, waveform, label, et in zip(ids, x, labels, event_type):
            et = et.decode("utf-8")
            if self.ramp > 0 and et != 'noise' :
                label = label_smoothing(label, self.f)
            if (et == 'noise' and self.include_noise) or et != 'noise':
                self.data.append({'x':waveform.astype(np.float32), 
                              'y':label.astype(np.float32), 
                              'metadata': None, 
                              'event_id': _id,
                              'et':et})
            
    def on_epoch_end(self):
        if self.file_buffer > 0:
            for file in choices(self.files, k=self.file_buffer):
                self._load_file(file)
                
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            shuffle(self.indexes)
            
    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))
    
    def __getitem__(self, item):
        ids = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]
        X, y = zip(*list(map(self.data_generation, ids)))
        y = np.stack(y, axis=0)
        y = np.split(y, y.shape[-1], axis=-1)
        X = np.stack(X, axis=0)
        return X, y
        
    def data_generation(self, _id):
        
        x = self.data[_id]['x'].copy()
        y = self.data[_id]['y'].copy()
        event_type = self.data[_id]['et']
        
        if self.augment:
            if event_type == 'noise':
                if np.random.random() < self.drop_channel:
                    x = waveform_drop_channel(x, np.random.choice(np.arange(x.shape[1])))
                if np.random.random() < self.add_gap:
                    x = waveform_add_gap(x, self.max_gap_size)
            
            else:
                if np.random.random() < self.add_event:
                    second = random.choice(self.data)
                    if second['et'] != 'noise':
                        second_x = second['x']
                        second_y = second['y']
                        
                        roll = np.random.randint(0, second_y.shape[0])
                        second_x = np.roll(second_x, roll, axis=0)
                        second_y = np.roll(second_y, roll, axis=0)
                        
                        scale = 1/np.random.uniform(1,10)
                        
                        second_x *= scale
                        second_y *= scale
                        
                        x += second_x
                        y = np.amax([y, second_y], axis=0)
                
                if np.random.random() < self.add_noise:
                    x = waveform_add_noise(x, np.random.uniform(0.01,0.15))
                if np.random.random() < self.drop_channel:
                    x = waveform_drop_channel(x, np.random.choice(np.arange(x.shape[1])))
                if np.random.random() < self.add_gap:
                    x = waveform_add_gap(x, self.max_gap_size)
        
        x, y = waveform_crop(x, y, self.new_length, self.testing)
        if self.taper_alpha > 0:
            x = waveform_taper(x, self.taper_alpha)
        
        if self.norm_mode is not None:
            x = waveform_normalize(x, mode=self.norm_mode, channel_mode=self.norm_channel_mode)
        
        x[np.isnan(x)] = self.fill_value
        y[np.isnan(y)] = self.fill_value
        
        return x, y
    
    
def create_class_weights(cw, y, eqt=False):
    if eqt:
        sw = np.where(y<0.2, cw[0], cw[1])
    else:
        sw = np.take(np.array(cw), np.argmax(y, axis=-1))
    return sw

class DropDetection(tf.keras.utils.Sequence):
    def __init__(self, 
                 super_sequence, 
                 p_classes, 
                 s_classes,
                 d_class=-1,
                 class_weights=None,
                 distance_weighting=False,
                 phasenet=True):
        self.super_sequence = super_sequence
        self.p_classes = p_classes
        self.s_classes = s_classes
        self.d_class = d_class
        self.phasenet = phasenet
        self.class_weights = class_weights
        self.distance_weighting = distance_weighting
       
    def __len__(self):
        return len(self.super_sequence)

    def __getitem__(self, idx):
        
        data = self.super_sequence.__getitem__(idx)
        if len(data) < 3:
            batch_x, batch_y = data
            metadata = None 
        else:
            batch_x, batch_y, metadata = data
            
        distance_weight = np.ones((len(batch_x), 1))

        p = np.concatenate([batch_y[i] for i in self.p_classes], axis=-1)
        s = np.concatenate([batch_y[i] for i in self.s_classes], axis=-1)
        d = batch_y[self.d_class]

        if not self.phasenet:
                p = np.max(p, axis=-1, keepdims=True)
                s = np.max(s, axis=-1, keepdims=True)
                y = [d, p, s]
                if not self.class_weights is None:
                    sw = [create_class_weights(self.class_weights, d, eqt=True), 
                          create_class_weights(self.class_weights, p, eqt=True),
                          create_class_weights(self.class_weights, s, eqt=True)]
                else:
                    sw = [np.ones((len(s),1,1)), np.ones((len(s),1,1)), np.ones((len(s),1,1))]
                
                sw = list(map(lambda s: np.expand_dims(distance_weight, axis=-1)*s, sw))
                
                if not metadata is None:
                    y.append(metadata)
                    sw.append(np.ones(len(metadata)))
            
        else:
                p = np.max(p, axis=-1, keepdims=True)
                s = np.max(s, axis=-1, keepdims=True)
                p = np.clip(p, 0, 1)
                s = np.clip(s, 0, 1)
                n = np.clip(1 - p - s, 0, 1)
                y = np.concatenate([n, p, s], axis=-1)
                if not metadata is None: 
                    y = [y, metadata]
                    sw = create_class_weights(self.class_weights, y) if not self.class_weights is None else np.ones((len(s),1))
                    sw = distance_weight * sw
                    sw = [sw, np.ones(len(metadata))]
                else:
                    sw = create_class_weights(self.class_weights, y) if not self.class_weights is None else np.ones((len(s),1))
                    sw = distance_weight * sw
        
        return batch_x, y, sw

    def on_epoch_end(self):
        self.super_sequence.on_epoch_end()

def angle_diff_tf(true,pred,sample_weight=None):
    true = tf.math.angle(tf.complex(true[:, 0], true[:, 1]))
    pred = tf.math.angle(tf.complex(pred[:, 0], pred[:, 1]))

    diff = tf.math.atan2(tf.math.sin(true-pred), tf.math.cos(true-pred))
    if sample_weight:
        return sample_weight * diff
    return diff

class AngleMAE(tf.keras.metrics.Metric):
    def __init__(self, name='angle_mse', **kwargs):
        super(AngleMAE, self).__init__(name=name, **kwargs)
        self.angle_mae = self.add_weight(name='amae', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        values = tf.math.abs(angle_diff_tf(y_true,y_pred))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.angle_mae.assign_add(tf.reduce_mean(values))
        self.count.assign_add(1)

    def result(self):
        return self.angle_mae / self.count
    

class IOU(tf.keras.metrics.Metric):
    def __init__(self, name='iou', threshold=0.5, **kwargs):
        super(IOU, self).__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.math.ceil(y_true - self.threshold)
        y_pred = tf.math.ceil(y_pred - self.threshold)
        res = tf.math.reduce_sum(tf.clip_by_value(y_true * y_pred, 0.0, 1.0), axis=(1,2))
        tot = tf.math.reduce_sum(tf.clip_by_value(y_true + y_pred, 0.0, 1.0), axis=(1,2))
        values = res/(tot + tf.keras.backend.epsilon())
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
            
        self.iou.assign_add(tf.reduce_mean(values))

    def result(self):
        return self.iou
    

class TruePositives(tf.keras.metrics.Metric):
    def __init__(self, 
                 name='tp',
                 dt=1,
                 **kwargs):
        super(TruePositives, self).__init__(name=name, **kwargs)
        self.dt = dt
        self.reset_state()
        
    def reset_state(self):
        self.data = []
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        n_dims = y_pred.shape[-1]
        
        if n_dims > 1:
            y_true = y_true[...,-2:]
            y_pred = y_pred[...,-2:]

        it = tf.math.argmax(y_true, axis=1)
        ip = tf.math.argmax(y_pred, axis=1)
            
        values = tf.where(abs(it-ip) < self.dt, tf.ones_like(it), tf.zeros_like(it))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
            
        self.data.append(tf.reduce_mean(values))
            
    def result(self):
        return tf.reduce_mean(self.data)
    
def recall_metric(dt=1.0):
    def recall(y_true, y_pred, sample_weight=None):
        n_dims = y_pred.shape[-1]
        
        if n_dims > 1:
            y_true = y_true[...,1:]
            y_pred = y_pred[...,1:]

        it = tf.math.argmax(y_true, axis=1)
        ip = tf.math.argmax(y_pred, axis=1)
            
        values = tf.where(abs(it-ip) < dt, tf.ones_like(it), tf.zeros_like(it))
        values = tf.cast(values, tf.float32)
        #if sample_weight is not None:
        #    sample_weight = tf.cast(sample_weight, tf.float32)
        #    sample_weight = tf.broadcast_to(sample_weight, values.shape)
        #    values = tf.multiply(values, sample_weight)
        return tf.reduce_mean(values)
    return recall

def kl_divergence_metric():
    def kld(y_true, y_pred, sample_weight=None):
        _, p, s = tf.split(y_true, 3, axis=-1)
        _, pt, st = tf.split(y_pred, 3, axis=-1)
        p = tf.squeeze(p)
        s = tf.squeeze(s)
        pt = tf.squeeze(pt)
        st = tf.squeeze(st)
        return (tf.keras.metrics.kl_divergence(p, pt) + tf.keras.metrics.kl_divergence(s, st)) / 2
    return kld

def js_divergence(p, q):
    pm = tf.reduce_sum(p, axis=-1, keepdims=True)
    pw = tf.where(pm < tf.keras.backend.epsilon(), tf.ones_like(pm), pm)
    qm = tf.reduce_sum(q, axis=-1, keepdims=True)
    qw = tf.where(qm < tf.keras.backend.epsilon(), tf.ones_like(qm), qm)
    p /= pm
    q /= qm
    m = (p + q) / 2
    return (tf.keras.metrics.kl_divergence(p, m) + tf.keras.metrics.kl_divergence(q, m) / 2)

def js_divergence_metric():
    def jsd(y_true, y_pred, sample_weight=None):
        _, p, s = tf.split(y_true, 3, axis=-1)
        _, pt, st = tf.split(y_pred, 3, axis=-1)
        p = tf.squeeze(p)
        s = tf.squeeze(s)
        pt = tf.squeeze(pt)
        st = tf.squeeze(st)
        return (js_divergence(p,pt) + js_divergence(s,st)) / 2
    return jsd

class CustomStopper(tf.keras.callbacks.EarlyStopping):
    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0, 
                 patience=0, 
                 verbose=0, 
                 mode='auto',
                 restore_best_weights=False, 
                 start_epoch=1): # add argument for starting epoch
        super(CustomStopper, self).__init__(monitor=monitor, 
                                            min_delta=min_delta, 
                                            patience=patience, 
                                            verbose=verbose, 
                                            restore_best_weights=restore_best_weights,
                                            mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
            
            
import tensorflow.keras.backend as K

class CategoricalFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, 
                 alpha=0.25, 
                 gamma=2.0, 
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='categorical_focal_crossentropy'):
        super(CategoricalFocalCrossentropy, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = self.alpha * y_true * K.pow((1-y_pred), self.gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=-1)
        return loss
        
        
def create_data_generator(files, config, training=True):
    if 'eqt' in config.model.type :
        include_noise = True
        phasenet = False
    else :
        include_noise = False
        phasenet = True
    phaselist = config.data.allowed_phases 
    p_classes = [i for i, e in enumerate(phaselist) if 'P' in e] #ALL SUBCLASSES OF P
    s_classes = [i for i, e in enumerate(phaselist) if 'S' in e] #ALL SUBCLASSES OF S
    d_class = [i for i, e in enumerate(phaselist) if 'D' in e][0] #DETECTION CLASS
    dataset = EQDatareader(files=files, 
                             new_length=int(config.data.sampling_rate*config.augment.new_size),
                             batch_size=config.training.batch_size,
                             taper_alpha=config.augment.taper,
                             add_noise=config.augment.add_noise,
                             add_event=config.augment.add_event,
                             drop_channel=config.augment.drop_channel,
                             add_gap=config.augment.add_gap,
                             max_gap_size=config.augment.max_gap_size,
                             augment=training,
                             norm_mode=config.normalization.mode,
                             norm_channel_mode=config.normalization.channel_mode,
                             shuffle=training,
                             testing=not training,
                             include_noise=include_noise,
                             ramp=config.augment.ramp)
    nchannels=dataset.data[0]['x'].shape[1]

    return DropDetection(dataset, 
                         p_classes=p_classes, 
                         s_classes=s_classes,
                         d_class=d_class,
                         class_weights=config.training.class_weights,
                         phasenet=phasenet),nchannels

def get_data_files(indir, years, config ):
    training_files=[]
    for y in years :
        training_files.extend(glob.glob(f'{indir}/traindata_center_only_{y}_*.hdf5'))
    training_files = list(filter(os.path.exists, training_files))
    return training_files

    
def get_model(config,nchannels):
    
    metadata_cols = []
    model_type = config.model.type
    
    input_shape = (int(config.data.sampling_rate*config.augment.new_size), nchannels)
    
    if config.model.probabilistic:
        import tensorflow_probability as tfp
        tfd = tfp.distributions
        tfpl = tfp.layers
        metadata_loss = 'mse'
        num_outputs = len(metadata_cols) + 1 #+1 for POLAR CORDS OF BAZ
        metadata_model = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tfpl.DenseFlipout(128, activation='relu'),
                tfpl.DenseFlipout(len(metadata_cols) + 1)])
    else:
        metadata_loss = 'mse'
        metadata_model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(128, activation='relu'),
                                            tf.keras.layers.Dense(len(metadata_cols) + 1)])
    filters = config.model.filters
    kernelsizes = config.model.kernel_sizes
    dropout = config.training.dropout
    pool_type = config.model.pooling_type
    try :
        activation = config.model.activation
    except ConfigAttributeError :
        activation = 'relu'

    use_metadata = len(metadata_cols) > 1

    opt = type(tf.keras.optimizers.get(config.training.optimizer))
    opt = opt(learning_rate=config.training.learning_rate, 
              weight_decay=config.training.weight_decay)
            
    kw = dict(num_classes=3,
                    dropout_rate=dropout,
                    filters=filters,
                    kernelsizes=kernelsizes,
                    pool_type=pool_type,
                    activation=activation, 
                    kernel_regularizer=tf.keras.regularizers.L1L2(config.training.l1_norm,
                                                                config.training.l2_norm),
                    output_activation='softmax')
        
    if model_type == 'phasenet':
        try :
            kw['conv_type'] = config.model.conv_type
        except ConfigAttributeError:
            kw['conv_type'] = 'default'

        model_class = nm.PhaseNet
        
    if model_type == 'transphasenet':
        kw['residual_attention'] = config.model.residual_attention
        kw['att_type'] = config.model.att_type
        kw['rnn_type'] = config.model.rnn_type
        kw['additive_att'] = config.model.additive_att
        model_class = nm.TransPhaseNet
        
    if model_type == 'epick':
        kw['residual_attention'] = config.model.residual_attention
        model_class = nm.EPick

    if not 'eqt' in model_type:
        def create_model():
            if use_metadata:
                model = model_class(metadata_model=metadata_model,
                                        ph_kw=kw)
                loss = [tf.keras.losses.CategoricalCrossentropy(), metadata_loss]
                loss_weights = config.model.loss_weights[:len(loss)]
                metrics = [[keras_f1], ['mse']]
            else:
                model = model_class(**kw)
                loss = tf.keras.losses.CategoricalCrossentropy()
                loss_weights = None
                metrics = [tf.keras.metrics.MeanMetricWrapper(lambda a,b: keras_f1(a[...,1], b[...,1]), name='f1_p'), 
                           tf.keras.metrics.MeanMetricWrapper(lambda a,b: keras_f1(a[...,2], b[...,2]), name='f1_s')]

            model.compile(optimizer=opt,
                        loss=loss,
                        loss_weights=loss_weights,
                        sample_weight_mode="temporal",
                        metrics=metrics)
            return model
        
    else:
        def create_model():
            try : resfilters=config.model.resfilters
            except ConfigAttributeError : resfilters = None
            kw = dict(input_dim=input_shape,
                    classify=True,
                    dropout=dropout,
                    filters=filters,
                    kernelsizes=kernelsizes,
                    activation=activation,
                    resfilters=resfilters,
                    reskernelsizes=config.model.res_kernel_sizes,
                    lstmfilters=config.models.lstm_filters,
                    transformer_sizes=config.models.transformer_sizes)
            model = nm.EarthQuakeTransformer(**kw)
            loss = [tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
                    tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
                    tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0)]
            metrics = None
            
            loss_weights = config.models.loss_weights[:len(loss)]
            
            model.compile(optimizer=opt,
                        loss=loss,
                        loss_weights=loss_weights,
                        sample_weight_mode=["temporal",
                                            "temporal",
                                            "temporal"],
                        metrics=metrics)

            return model
        
    return create_model()


def get_predictions(config, test_dataset, model):
        
    #PREDICT
    true = []
    pred = []
    meta = []
    metat_mean = []
    metat_std = []
    
    use_metadata = False

    sample_weight = []
    xte = []
    
    for x, y, sw in tqdm(test_dataset):
        xte.append(x)
        if not 'eqt' in config.model.type:
            if use_metadata:
                y, m = y
            else:
                m = np.zeros((y.shape[0],))
            if config.model.probabilistic:
                tmp_p = []
                tmp_m = []
                for _ in range(config.model.probabilistic_samples):
                    a = model.predict_on_batch(x)
                    if use_metadata:
                        p, mt = a
                    else:
                        p = a
                        mt = np.zeros((p.shape[0],))
                    tmp_m.append(mt)
                    tmp_p.append(p)
                mt = np.asarray(tmp_m)
                mt_mean = np.mean(mt, axis=0)
                mt_std = np.std(mt, axis=0)
                p = np.mean(np.asarray(tmp_p), axis=0)
            else:
                a = model.predict_on_batch(x)
                if use_metadata:
                    p, mt_mean = a
                else:
                    p = a
                    mt_mean = np.zeros((p.shape[0],))
                mt_std = np.zeros_like(mt_mean)
        else:
            if use_metadata:
                dd, dp, ds, m = y
            else:
                dd, dp, ds = y
                m = np.zeros((dd.shape[0],))
                
            y = np.stack([dd,dp,ds], axis=-1)
            if config.model.probabilistic:
                tmp_d, tmp_p, tmp_s = [], [], []
                tmp_m = []
                for _ in range(config.training.probabilistic_samples):
                    a = model.predict_on_batch(x)
                    if use_metadata:
                        d, p, s, mt = a
                    else:
                        d, p, s = a
                        mt = np.zeros((d.shape[0], ))
                        
                    tmp_m.append(mt)
                    tmp_d.append(d)
                    tmp_p.append(p)
                    tmp_s.append(s)
                mt_mean = np.mean(np.asarray(tmp_m), axis=0)
                mt_std = np.std(np.asarray(tmp_m), axis=0)
                d = np.mean(np.asarray(tmp_d), axis=0)
                p = np.mean(np.asarray(tmp_p), axis=0)
                s = np.mean(np.asarray(tmp_s), axis=0)
            else:
                a = model.predict_on_batch(x)
                if use_metadata:
                    d, p, s, mt_mean = a
                else:
                    d, p, s = a
                    mt_mean = np.zeros((d.shape[0],))
                mt_std = np.zeros_like(mt_mean)
            p = np.stack([d,p,s], axis=-1)

        meta.append(m)
        metat_mean.append(mt_mean)
        metat_std.append(mt_std)
        
        sample_weight.append(np.asarray(sw))
        pred.append(p)
        true.append(y)
    xte, true, pred, meta, metat_mean, metat_std, sample_weight = map(lambda x: np.concatenate(x, axis=0), 
                                                                  [xte, true, pred, meta, metat_mean, metat_std, sample_weight])

    return xte, true, pred, meta, metat_mean, metat_std, sample_weight 
