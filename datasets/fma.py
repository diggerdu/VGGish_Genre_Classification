import os
import os.path as osp
import sys

import numpy as np
import torch
import h5py
from tqdm import tqdm

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import musdb
from yacs.config import CfgNode as CN
import SharedArray as sa
import random
import datasets.fma_utils as utils
from datasets import vggish_input
import pandas as pd
import librosa as lb
import pathos.multiprocessing as pmp


def norm(arr, thres=None, max_=None):
    arr -= np.mean(arr)
    if thres is not None and np.max(arr) < thres:
        arr = arr
        silence = True
    else:
        if max_ is not None:
            arr = arr / max_
        else:
            arr = arr / float(np.max(arr))
        silence = False
    return arr, silence

def norm_std(arr, thres=None, std=None):
    arr -= np.mean(arr)
    if thres is not None and np.max(arr) < thres:
        arr = arr
        silence = True
    else:
        if std is not None:
            arr = arr / std
        else:
            arr = arr / float(np.std(arr))
        silence = False
    return arr, silence

class h5_wrapper:
    def __init__(self, h5fn):
         self.h5fn = h5fn
         h5_file = h5py.File(self.h5fn, 'r', libver='latest', swmr=True)
         self.keys = list(h5_file.keys())
         h5_file.close()
         print("DATA SOURCE INFO ", h5fn, "LEN ", len(self.keys))
    def __getitem__(self, index):
        return self._get_from_h5(index)     
    def __len__(self):
        return len(self.keys)
    def _get_from_h5(self, index):
        h5_file = h5py.File(self.h5fn, 'r', libver='latest', swmr=True)
        ele = h5_file[self.keys[index]]
        #audio = np.array(ele)
        feat = np.array(ele)
        sr = ele.attrs['sr']
        label = ele.attrs['label']
        return dict(sr=sr, 
                #audio=audio, 
                feat=feat,
                label=label)
    




sb = ['Easy Listening', 'Jazz', 'Blues', 'Instrumental', 'Old-Time / Historic', 'Soul-RnB', 'Hip-Hop', 'Electronic', 'Experimental', \
'Spoken', 'International', 'Classical', 'Folk', 'Rock', 'Pop', 'Country']

def make_dataset(data_path, cfg, mode='training'):
    try:
        assert cfg.serialized

        return h5_wrapper(cfg.dump_path)
    except:
        track_pd = pd.read_csv(os.path.join(data_path, "fma_metadata/tracks.csv"), index_col=0, header=[0, 1])
        #track_pd = track_pd[track_pd.set.subset == 'small']
        track_pd = track_pd[track_pd.set.split == mode]
        track_pd = track_pd[track_pd.track.genre_top.notnull()]
        pair_list = list()
        for id_, info in tqdm(track_pd.iterrows(), total= len(track_pd.index)):
            tid_str = '{:06d}'.format(id_)
            audio_path = os.path.join(data_path, "fma_medium", tid_str[:3], tid_str + '.mp3')
            pair_list.append((audio_path, info.track.genre_top)) 
              

        p = pmp.ProcessPool(96)
        proc_fn = _proc_func()
        data_iter = p.imap(proc_fn, pair_list)
        with tqdm(total=len(pair_list)) as pbar:
            for i, res in enumerate(tqdm(data_iter)):
                pbar.update()
                if cfg.dump_path is not None and cfg.need_serialize:
                    if res is not None:
                        dump_to_h5(cfg.dump_path, res)

        return h5_wrapper(cfg.dump_path)

def dump_to_h5(dump_path, element):
    f = h5py.File(dump_path, "a" if osp.isfile(dump_path) else 'w', libver="latest")
    audio = element['audio']
    if np.isnan(audio).sum() > 0 or np.isinf(audio).sum() > 0:
        __import__('pdb').set_trace()
    k = osp.basename(element['audio_fn'])
    f.create_dataset(k, data=element['feat'])
    f[k].attrs['sr'] = element['sr']
    f[k].attrs['audio_fn'] = element['audio_fn']
    f[k].attrs['label'] = element['label']
    f.close()



def _proc_func():
    def _proc(fn_pair):
        audio_fn, label = fn_pair
        try:
            audio, sr = lb.load(audio_fn, sr=44100, mono=True)
            feat = vggish_input.waveform_to_examples(audio, sr)
        except:
            #print(audio_fn, " 文件损坏")
            return None
        return dict(audio=audio, sr=sr, audio_fn=audio_fn, label=label, feat=feat)
    return _proc


class fma(data.Dataset):
    def __init__(self, config):
        self.mode = config.mode
        print(f'########subset:{self.mode}###########')
        self.data = make_dataset(config.data_root, config, mode=self.mode)

    def __getitem__(self, index):
        data_dict = self.data[index]
        #audio = data_dict['audio']
        sr = data_dict['sr']
        label = sb.index(data_dict['label'])
        assert len(sb) == 16
        assert label < len(sb)
        #feat = vggish_input.waveform_to_examples(audio, sr)
        feat = data_dict['feat']
        if feat.shape[0] > 32:
            feat = feat[:32,:,:]
        else:
            gap = 32 - feat.shape[0]
            pad_w = ((0, gap),(0,0),(0,0))
            feat = np.pad(feat, pad_width=pad_w, mode='constant')
   
        res = dict(feat=feat,
                    label=label
                    ) 
        return res

    def __len__(self):
        return len(self.data)




class FmaDB:
    def __init__(self, config):
        print('#####LOADING DATASOURCE#####')
        self.config = config
        train_set = fma(config.train)
        test_set = fma(config.test)
        self.mode = 'Train'

        distributed = self.config.distributed
        train_sampler = None
        test_sampler = None
        world_size = 1
        if distributed:
            local_rank = int(sys.argv[1].split('=')[-1])
            world_size = torch.distributed.get_world_size()
            print('######world size {}#####'.format(world_size))
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=world_size, rank=local_rank)


        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=(train_sampler is None),
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory, sampler=train_sampler)

        self.test_loader = DataLoader(test_set, batch_size=self.config.test.batch_size, shuffle=(test_sampler is None),
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory, sampler=test_sampler)

        self.train_iterations = len(train_set) // self.config.batch_size
        self.test_iterations = len(test_set) // self.config.test.batch_size  // world_size

        self.loader = self.train_loader
        self.iterations = self.train_iterations // world_size
        self.counter = 0

if __name__ == "__main__":
    config = CN()
    config.batch_size = 12
    config.data_loader_workers = 24
    config.pin_memory = True 
    config.train = CN()
    config.train.data_root = "/root/thome/musdb18_wav"
    config.train.sample_dis = 1024
    config.train.segLen = 44100 * 7
    config.train.mode = "Train"

    # make_dataset('Train', "/root/thome/musdb18/train_data_unet.h5")
    
    sb = Musdb18DB(config)
    for i in sb.train_loader:
        __import__('pdb').set_trace() 

    __import__('pdb').set_trace() 
