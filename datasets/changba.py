import sys
import torch
import librosa as lb
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
import pyarrow as pa
import pyarrow.plasma as plasma
import random
import apex

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import musdb
from yacs.config import CfgNode as CN
import os
import os.path as osp
from .norm import *

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
        '''
        try:
            h5_file = h5py.File(cfg.dump_path, 'r', libver='latest', swmr=True)
            fn_list = list(h5_file.keys())
            audio_fn_list = list(set(audio_fn_list) - set(fn_list))
        except:
            pass
        '''

        self.note_fn_list, self.audio_fn_list = self._match_pair(sorted(glob(cfg.note_pattern)), audio_fn_list)
        if cfg.dump_path is not None and cfg.serialized:
            #self.data = h5py.File(cfg.dump_path, 'r', libver='latest', swmr=True)
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
                        dump_to_h5(cfg.dump_path, res)
        #self.data = h5py.File(cfg.dump_path, 'r', libver='latest', swmr=True)
        

    def __getitem__(self, index):
        cfg = self.config
        self.data = h5py.File(cfg.dump_path, 'r', libver='latest', swmr=True)
        start = cfg.sample_dis * index
        end = start + cfg.seg_len
        assert end <= self.data['wav'].shape[0]

        mix_wav = self.data["wav"][start:end]
        mix_wav, silence = norm(mix_wav, thres=0.02)

        pitch_seq = self.data['pitch_ref'][start:end]
        pitch_ref = lb.util.frame(pitch_seq, cfg.nfft, cfg.hop_len).mean(axis=0).astype(int)
        assert pitch_ref.shape[0] == cfg.n_frames


        res =  {"mix_wav": np.repeat(mix_wav[::,np.newaxis], 2, axis=-1), \
        "pitch_ref": pitch_ref,
        "silence": silence
        }
        self.data.close()
        return res

    def __len__(self):
        cfg = self.config
        self.data = h5py.File(cfg.dump_path, 'r', libver='latest', swmr=True)
        len_ = (self.data["wav"].shape[0] - self.config.seg_len) // self.config.sample_dis
        self.data.close()
        return len_

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

        ##TODO padding casued mismatch
        wav_length = audio.shape[0]
        #n_frames = (wav_length - cfg.nfft) // cfg.hop_len + 1
        #wav_length = (n_frames - 1) * cfg.hop_len + cfg.nfft
        
        #audio = audio[:wav_length]
        audio -= np.mean(audio)
        audio /= np.max(audio)

        if note_fn is not None:
            try:
                note_mat = np.loadtxt(note_fn)
                pitch_time = np.arange(wav_length) / cfg.sr * 1000. #millisecond
                pitch_ref = np.zeros(wav_length, dtype=np.int8)
                pitch_ref_ori = np.zeros(wav_length, dtype=np.int8) 

                ## acapllea range from #30 #90
                ## add up/lower bounds to config file
                for i in range(note_mat.shape[0]):
                    t0 = note_mat[i, 0]
                    t1 = (note_mat[i, 0] + note_mat[i, 1]) 
                    pv_ori = int(note_mat[i, 2])
                    pv = pv_ori - 29
                    if pv >= 61:
                        pv = 61
                    elif pv <= 0:
                        pv = 0
                    pitch_ref[(pitch_time >= t0) & (pitch_time <= t1)] = pv
                    #pitch_ref_ori[(pitch_time >= t0) & (pitch_time <= t1)] = pv_ori
            except:
                print(note_fn)
                return None
        else:
            pitch_ref = None
            pitch_ref_ori = None
            

        element = {"wav": audio.astype(np.float32),
                    "pitch_ref":pitch_ref,
                    "audio_fn":audio_fn,
                    "note_fn":note_fn,
                    }
        return element

    return _proc

def dump_to_h5(dump_path, element):
    f = h5py.File(dump_path, "a" if osp.isfile(dump_path) else 'w', libver="latest")
    
    audio = element['wav']
    if np.isnan(audio).sum() > 0 or np.isinf(audio).sum() > 0:
        __import__('pdb').set_trace() 
    for k in ["wav", "pitch_ref"]:
        data = element[k]
        if not k in list(f.keys()):
            f.create_dataset(k, data=data, maxshape=(None, ), chunks=True, dtype=data.dtype)
        else:
            f[k].resize((f[k].shape[0] + data.shape[0],))
            f[k][-1*data.shape[0]:] = data

        debug = f[k][-1*data.shape[0]:]
        if np.isnan(audio).sum() > 0:
            __import__('pdb').set_trace()
        if np.isinf(audio).sum() > 0:
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
