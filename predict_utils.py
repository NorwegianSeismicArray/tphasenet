# Copyright 2023 Andreas Koehler, Erik Myklebust, MIT license
# Code for making prediction with phase detection models: contineous or window-wise


import numpy as np
from obspy import UTCDateTime, Stream, Trace
from scipy.signal.windows import tukey
import tensorflow as tf
import scipy as sp
from tqdm import tqdm
import sys
from scipy.signal import find_peaks
import glob

class LivePhaseNet:
    def __init__(self,
                 model,
                 client,
                 station='ARA0',
                 channels="HHZ,HHE,HHN,BHZ,BHE,BHN,sz,se,sn,bz,be,bn",
                 length=60,
                 step=10,
                 bandpass=None,
                 delay=10,
                 return_raw=False,
                 verbose=True,
                 normalization='max',
                 eqt=False,
                 normalization_mode='global',
                 stream=None, # to speed up we can pre-load stream and create windows by slicing
                 taper=0.01,
                 sampling_rate=40):
        self.model = model
        self.client = client
        self.eqt = eqt
        self.return_raw = return_raw
        self.station = station
        self.channels = channels
        self.length = length
        self.delay = delay
        self.step = step
        self.bandpass = bandpass
        self.normalization = normalization
        self.normalization_mode = normalization_mode
        self.verbose = verbose
        self.sampling_rate = sampling_rate
        self.taper = taper
        #remove double components stream
        if len(stream.copy().select(station=station)) != 3 :
            for tr in stream.copy().select(station=station)[3:] : stream.remove(tr)
        self.stream = stream
        self.w = tukey(int(length*sampling_rate), taper)[np.newaxis,:,np.newaxis]

    def _normalize(self, X, mode='max', channel_mode='local'):
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

    def _load_data(self, start, end,cut=True):
        length = int(self.length * self.sampling_rate)
        if start is None or end is None:
            start, end = UTCDateTime.now() - self.length - self.delay, UTCDateTime.now() - self.delay
        if self.stream is None :
            st_org = self.client.get_waveforms('*',self.station,'*', self.channels, start, end)
        else : st_org=self.stream.slice(start, end)
        st_org.sort()
        if len(st_org) == 0 : return
        if len(st_org) > 3 : st_org = st_org[:3]
        st_org.detrend()
        st_org.taper(self.taper)
        if self.bandpass is not None:
            st_org.filter('bandpass', freqmin=self.bandpass[0], freqmax=self.bandpass[1])
        for trace in st_org:
            sr = trace.stats.sampling_rate
            break
        if sr != self.sampling_rate:
            st_org.resample(self.sampling_rate, no_filter=True)
        if cut :
            data = np.stack([trace.data[:length] for trace in st_org], axis=-1)
            assert len(data) == length
        else : data = np.stack([trace.data for trace in st_org], axis=-1)
        return data

    def predict(self, start=None, end=None, step=None):
        delta = end - start
        raw = [self._load_data(start + s, start + s + self.length) for s in np.arange(0, delta, step)]
        x = [self._normalize(r, self.normalization, self.normalization_mode) for r in raw]
        x = np.stack(x, axis=0) * self.w
        
        v = 1 if self.verbose else 0
        if self.eqt:
            d, p, s = self.model.predict(x, verbose=v)#, batch_size=8)
            prediction = np.stack([p,s,d], axis=-1)
        else:
            prediction = self.model.predict(x, verbose=v)

        prediction = np.squeeze(prediction)
        
        if self.return_raw:
            return np.expand_dims(raw, axis=0), prediction
        else:
            return x, prediction

def phase_detection(client,cfg_pred,cfg_model):

    model_name = cfg_pred.modelname
    model_type = cfg_pred.modeltype
    # contineous mode, do hour-wise predictions #
    if cfg_pred.cont_processing :
        times = []
        start_cont = UTCDateTime(cfg_pred.start_time)
        end_cont = UTCDateTime(cfg_pred.end_time)
        window = [0,cfg_pred.window_length]
        t = start_cont
        while t+cfg_pred.window_length <= end_cont:
            times.append(t)
            t += cfg_pred.window_length
    else :
        times = list(map(UTCDateTime, times))
        # before and after event
        window = [cfg_pred.window_offset,cfg_pred.window_length-cfg_pred.window_offset]

    model = tf.keras.models.load_model(f'{model_name}_{model_type}.tf', compile=False)

    statlist = cfg_pred.stations

    channels= 'HHZ,HHE,HHN,BHZ,BHE,BHN,SHZ,SHE,SHN,BH1,BH2,sz,se,sn,bz,be,bn'
    folder = cfg_pred.output_dir
    length = model.layers[0].input.shape[1] // cfg_model.data.sampling_rate

    pt = {}
    st = {}
    for station in statlist :
        pt[station]=[]
        st[station]=[]
    for event_time in tqdm(times):
        event_time_stripped = str(event_time.isoformat().split('.')[0].replace(':','')).strip()
        start = event_time - window[0]
        end = event_time + window[1]

        for station in statlist :
            print('Initializing')
            prediction = LivePhaseNet(model,
                      client,
                      station=station,
                      channels=channels,
                      length=length,
                      return_raw=True,
                      eqt=False,
                      normalization=cfg_model.normalization.mode,
                      normalization_mode=cfg_model.normalization.channel_mode,
                      verbose=True,
                      sampling_rate=cfg_model.data.sampling_rate,
                      taper=cfg_model.augment.taper,
                      stream=client.get_waveforms('*',station,'*', channels, start, end+length),
                      bandpass=(cfg_model.data.lower_frequency, cfg_model.data.upper_frequency))

            print("Predicting for time window and station:",start,end,station)
            #try:
            if True :
                _, pred = prediction.predict(start, end, cfg_pred.step)
            #except ValueError as e:
            #    print(e)
            #    continue

            print("Combining output ...")
            sr = cfg_model.data.sampling_rate
            total = ((end - start) * sr) + len(pred[0])
            pred_padded = []
            for i,p in enumerate(pred):
                before = int(i*cfg_pred.step*sr)
                after = int(total - len(p) - before) - int(cfg_pred.step*sr)
                if i == 0 :
                    pred_padded.append(np.concatenate((p,np.array([[np.NAN,np.NAN,np.NAN]
                                                   for a in range(after)])),axis=0))
                elif after<1 :
                    pred_padded.append(np.concatenate((np.array([[np.NAN,np.NAN,np.NAN]
                                                   for b in range(before)]),p),axis=0))
                else :
                    pred_padded.append(np.concatenate((np.array([[np.NAN,np.NAN,np.NAN]
                                   for b in range(before)]),p,np.array([[np.NAN,np.NAN,np.NAN]
                                                      for a in range(after)])),axis=0))
            pred_padded = np.ma.masked_invalid(pred_padded)
            if cfg_pred.stacking == 'std' : pred_padded=np.ma.std(pred_padded,axis=0)
            if cfg_pred.stacking == 'median' : pred_padded=np.ma.median(pred_padded,axis=0)
            if cfg_pred.stacking == 'p25' : pred_padded=np.ma.masked_invalid(
                                  np.nanpercentile(np.ma.filled(pred_padded,np.nan),25,axis=0))
            if cfg_pred.stacking == 'mean' : pred_padded=np.ma.mean(pred_padded,axis=0)
        
            print("Saving output ...")
            t = np.arange(int((cfg_pred.window_length+prediction.length)*cfg_model.data.sampling_rate))
            t = [start + i for i in t / cfg_model.data.sampling_rate]
            if cfg_pred.save_waveforms :
                np.savez(f'{folder}/{model_type}_{station}_{cfg_pred.stacking}_{event_time_stripped}.npz',
                     t=t,
                     x=prediction._load_data(start, end+prediction.length, cut=False)[:-1],
                     y=pred_padded)
            if cfg_pred.save_prob :
                np.savez(f'{folder}/{model_type}_{station}_{cfg_pred.stacking}_{event_time_stripped}.npz',
                 t=t,
                 y=pred_padded)

            if cfg_pred.save_picks :
                d, p, s = np.split(pred_padded, 3, axis=1)
                p = find_peaks(np.squeeze(p),cfg_pred.p_threshold,
                    distance=int(cfg_model.data.sampling_rate*0.5))[0]
                s = find_peaks(np.squeeze(s),cfg_pred.s_threshold,
                    distance=int(cfg_model.data.sampling_rate*0.5))[0]
                if len(p) > 0:
                    for i in p: pt[station].append(t[i])
                if len(s) > 0:
                    for i in s: st[station].append(t[i])
                # save what we have so far. Will be overwritten with new time windows.
                np.savez(f'{folder}/{model_type}_{station}_{cfg_pred.stacking}_picks.npz',
                     p=pt[station],
                     s=st[station])
