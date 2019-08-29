import os

import numpy as np
import torch
import h5py
from tqdm import tqdm

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import musdb
from yacs.config import CfgNode as CN



def make_dataset(data_path, mode=None):
    mus = musdb.DB(root=data_path, is_wav=True, subsets="test")
    mixture_list = list() 
    vocal_list = list()
    for track in tqdm(mus):
        mixture_list.append(track.audio.sum(axis=-1))
        vocal_list.append(track.targets['vocals'].audio.sum(axis=-1))
    mixture_array = np.concatenate(mixture_list)
    vocal_array = np.concatenate(vocal_list)
    assert mixture_array.shape == vocal_array.shape

    return dict(mixture_array=mixture_array,
                vocal_array=vocal_array
                )

class musdb18(data.Dataset):
    def __init__(self, config):
        self.data_dicts = make_dataset(config.data_root)
        self.mode = config.mode
        self.sample_dis = config.sample_dis
        self.segLen = config.segLen

    def __getitem__(self, index):
        start = self.sample_dis * index
        end = start + self.segLen
        return self.data_dicts["mixture_array"][start:end], self.data_dicts["vocal_array"][start:end]

    def __len__(self):
        return (self.data_dicts["mixture_array"].shape[0] - self.segLen) // self.sample_dis


class Musdb18DB:
    def __init__(self, config):
        self.config = config
        train_set = musdb18(config.train)
        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
        self.train_iterations = len(train_set) // self.config.batch_size

    def finalize(self):
        pass

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
