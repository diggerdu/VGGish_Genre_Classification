import sys
import torch
import librosa as lb
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
#import hickle as hkl
#import pyarrow as pa
#import pyarrow.plasma as plasma
#import random
#import apex

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import musdb
from yacs.config import CfgNode as CN
import os
import os.path as osp

# from profile_wrapper import *

# replace pickle with dill for dump local function
import pathos.multiprocessing as pmp



# @cprofile_wrapper("h5_wrapper.prof")
class h5_wrapper:
    def __init__(self, h5fn, cache_path=None):
        self.h5fn = h5fn
        h5_file = h5py.File(self.h5fn, 'r', libver='latest', swmr=True)
        self.keys = list(h5_file.keys())
        h5_file.close()
        self.cache_path = cache_path
        print("DATA SOURCE INFO ", h5fn, "LEN ", len(self.keys))
    def __getitem__(self, index, pre_cache=False):
        if self.cache_path is not None:
            plasma_client = plasma.connect(self.cache_path)
            offset = 52425
            obj_id = plasma.ObjectID((index+offset).to_bytes(20, byteorder="little"))
            if plasma_client.contains(obj_id):
                ele = plasma_client.get(obj_id)
                print('#######HIT##########')
            else:
                ele = self._get_from_h5(index)
                #from remote_pdb import RemotePdb
                #from random import randint
                #RemotePdb('127.0.0.1', 4444+randint(0,1000)).set_trace()
                if pre_cache:
                    plasma_client.put(ele, object_id=obj_id)
            plasma_client.disconnect()
        else:
            ele = self._get_from_h5(index)

        return ele

    def __len__(self):
        return len(self.keys)

    def _get_from_h5(self, index):
        h5_file = h5py.File(self.h5fn, 'r', libver='latest', swmr=True)
        ele_bytes = h5_file[self.keys[index]][()].tobytes()
        h5_file.close()

        try: 
            ele = pa.deserialize(ele_bytes)
        except:
            print('@@@@@@@@@', index, len(ele_bytes))
            print(self.h5_file)
        return ele

    def pre_cache(self):
        print('#####pre_cache####')
        for i in tqdm(range(len(self))):
            self.__getitem__(i, pre_cache=True)

class changba(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.initialize()

    def initialize(self):
        cfg = self.config
        self.mode = cfg.mode
        proc_fn = _proc_func(self.config)
        self.proc_fn = proc_fn
        audio_fn_list = glob(cfg.audio_pattern)
        try:
            h5_file = h5py.File(self.h5fn, 'r', libver='latest', swmr=True)
            fn_list = list(h5_file.keys())
            audio_fn_list = list(set(audio_fn_list) - set(fn_list))
        except:
            pass

        self.note_fn_list, self.audio_fn_list = self._match_pair(sorted(glob(cfg.note_pattern)), audio_fn_list)
        if cfg.dump_path is not None and cfg.serialized:
            self.data = h5_wrapper(cfg.dump_path, cfg.cache_path)
            return None


        #profile
        p = pmp.ProcessPool(cfg.num_workers)
        data_iter = p.imap(proc_fn, zip(self.audio_fn_list, self.note_fn_list))
        self.data = list()
        with tqdm(total=len(self.audio_fn_list)) as pbar:
            for i, res in enumerate(tqdm(data_iter)):
                pbar.update()
                if cfg.dump_path is not None and cfg.need_serialize:
                    if res is not None:
                        dump_to_h5(cfg.dump_path, res, res['audio_fn'])
        self.data = h5_wrapper(cfg.dump_path)
        

    def __getitem__(self, index):
        cfg = self.config
        sample_index = index // len(self.data)
        fn_index = index % len(self.data)

        os.system(f"echo {index} {fn_index} {sample_index} >> /tmp/log")
        sample = self.data[fn_index] 
        sample_len = sample['n_frames']
        step = (sample_len - cfg.n_frames) // cfg.num_per_file
        start_idx = step * sample_index

        cqt = sample["cqt"]
        pitch_ref = sample["pitch_ref"]
        pitch_ref_ori = sample["pitch_ref_ori"]
        
        res =  {"cqt": cqt[start_idx:start_idx+cfg.n_frames], \
        "pitch_ref": pitch_ref[start_idx:start_idx+cfg.n_frames],
        "pitch_ref_ori": pitch_ref_ori[start_idx:start_idx+cfg.n_frames],
        }
        return res

    def __len__(self):
        return len(self.data) * self.config.num_per_file

    def _match_pair(self, note_list, audio_list):
        proc_fn = lambda fn:osp.basename(fn).split('.')[0].split('_')[0]
        audio_dict = dict(zip(list(map(proc_fn, audio_list)), audio_list))
        note_dict = dict(zip(list(map(proc_fn, note_list)), note_list))
        id_list = list(set(audio_dict.keys()).intersection(note_dict.keys()))
        audio_list = [audio_dict[id_] for id_ in id_list]
        note_list = [note_dict[id_] for id_ in id_list]
        return note_list, audio_list

def _proc_func(cfg):
    def _proc(fn_pair):
        audio_fn, note_fn = fn_pair
        try:
            audio, _ = lb.load(audio_fn, sr=cfg.sr, mono=True)
        except:
            print(audio_fn, " 文件损坏")
            return None
        audio /= np.std(audio)
        audio -= np.mean(audio)

        ##TODO padding casued mismatch
        cqt = lb.cqt(audio, sr=cfg.sr, hop_length=cfg.hop_len, fmin=55.0, n_bins=301, bins_per_octave=60)
        cqt = np.abs(cqt).transpose()
        n_frames = cqt.shape[0]
        
        if note_fn is not None:
            try:
                note_mat = np.loadtxt(note_fn)
                pitch_time = np.arange(n_frames) * cfg.hop_len / cfg.sr * 1000. #millisecond
                pitch_ref = np.ones(n_frames, dtype=int) * 61
                pitch_ref_ori = np.zeros(n_frames, dtype=int) 

                ## acapllea range from #30 #90
                ## add up/lower bounds to config file
                for i in range(note_mat.shape[0]):
                    t0 = note_mat[i, 0]
                    t1 = (note_mat[i, 0] + note_mat[i, 1]) 
                    pv_ori = int(note_mat[i, 2])
                    pv = pv_ori - 30
                    if pv >= 61:
                        pv = 60
                    elif pv < 0:
                        pv = 61
                    pitch_ref[(pitch_time >= t0) & (pitch_time <= t1)] = pv
                    pitch_ref_ori[(pitch_time >= t0) & (pitch_time <= t1)] = pv_ori
            except:
                print(note_fn)
                return None
        else:
            pitch_ref = None
            pitch_ref_ori = None
            

        element = {"cqt": cqt,
                    "pitch_ref":pitch_ref, 
                    "pitch_ref_ori":pitch_ref_ori,
                    "n_frames":n_frames,
                    "audio_fn":audio_fn,
                    "note_fn":note_fn
                    }
        return element

    return _proc

def dump_to_h5(dump_path, element, ds_key):
    ds_key = str(ds_key)
    ele_bytes = pa.serialize(element).to_buffer().to_pybytes()
    f = h5py.File(dump_path, "a" if osp.isfile(dump_path) else 'w', libver="latest")
    
    try:
        f.create_dataset(str(ds_key), data = np.void(ele_bytes))
    except:
        pass

    if False: #for debug
        try:
            assert f[ds_key][()].tobytes() == ele_bytes
        except:
            __import__('pdb').set_trace() 
    f.close()

       


class ChangbaDB:
    def __init__(self, config):
        print('#####LOADING DATASOURCE#####')
        self.config = config
        train_set = changba(config.train) 
        test_set = changba(config.test)
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
    config.train.audio_pattern = "/root/thome/data/lab_aligned_data/*.mp3"
    config.train.note_pattern = "/root/thome/data/lab_aligned_data/*.note"
    config.train.dump_path = "/root/thome/data/melody_extraction/changba.h5"
    config.train.serialized = True
    config.train.need_serialize = True

    config.train.sr = 44100 
    config.train.hop_len = 512
    config.train.mode = "Train"
    config.train.num_workers = 16
    config.train.n_frames = int(24. * config.train.sr / config.train.hop_len)
    config.train.num_per_file = int((4 * 60 / 24) * 2)

    # make_dataset('Train', "/root/thome/musdb18/train_data_unet.h5")
    

    sd = h5_wrapper(config.train.dump_path)
    
    import time
    start = time.time()
    for i in range(10):
        sd[random.randint(0, 3000)]
    print(time.time() - start)


    sb = ChangbaDB(config)
    __import__('pdb').set_trace() 
    sb.train_loader.dataset.__getitem__(100)
    __import__('ipdb').set_trace() 
    for i in sb.train_loader:
        __import__('pdb').set_trace() 

    __import__('pdb').set_trace() 
