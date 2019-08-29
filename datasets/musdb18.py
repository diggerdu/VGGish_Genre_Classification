import os
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


def make_dataset(data_path, mode=None):
    try:
        mixture_array = sa.attach(f"shm://{mode}_mixture_array")
        vocal_array = sa.attach(f"shm://{mode}_vocal_array")

    except:
        mus = musdb.DB(root=data_path, is_wav=True, subsets=mode)
        mixture_list = list() 
        vocal_list = list()
        for track in tqdm(mus):
            #mixture_list.append(track.audio.sum(axis=-1))
            mixture_list.append(norm(track.audio)[0])
            #vocal_list.append(track.targets['vocals'].audio.sum(axis=-1))
            vocal_list.append(norm(track.targets['vocals'].audio)[0])
        mixture_array = np.concatenate(mixture_list)
        vocal_array = np.concatenate(vocal_list)

        assert mixture_array.shape == vocal_array.shape

        mixture_array_sa = sa.create(f"shm://{mode}_mixture_array", mixture_array.shape)
        vocal_array_sa = sa.create(f"shm://{mode}_vocal_array", vocal_array.shape)
        mixture_array_sa[::] = mixture_array
        vocal_array_sa[::] = vocal_array

    return dict(mixture_array=mixture_array,
                vocal_array=vocal_array
                )

class musdb18(data.Dataset):
    def __init__(self, config):
        self.mode = config.mode
        print(f'########subset:{self.mode}###########')
        self.data_dicts = make_dataset(config.data_root, mode=self.mode)
        self.sample_dis = config.sample_dis
        self.seg_len = config.seg_len

    def __getitem__(self, index):
        start = self.sample_dis * index
        end = start + self.seg_len
        mix_wav = self.data_dicts["mixture_array"][start:end]
        vocal_wav = self.data_dicts["vocal_array"][start:end]

        mix_std = np.std(mix_wav)
        mix_max = np.max(mix_wav)
        #mix_wav, silence = norm_std(mix_wav, thres=1e-2)
        mix_wav, silence = norm(mix_wav, thres=1e-2)

        if silence:
            return self.__getitem__(random.randint(0, len(self)-1))
        #vocal_wav, _  = norm_std(vocal_wav, thres=1e-2, std=mix_std)
        vocal_wav, _  = norm(vocal_wav, thres=1e-2, max_=mix_max)
   
        return dict(mix_wav=mix_wav,
                    vocal_wav=vocal_wav
                    ) 

    def __len__(self):
        return (self.data_dicts["mixture_array"].shape[0] - self.seg_len) // self.sample_dis




class Musdb18DB:
    def __init__(self, config):
        print('#####LOADING DATASOURCE#####')
        self.config = config
        train_set = musdb18(config.train)
        test_set = musdb18(config.test)
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
