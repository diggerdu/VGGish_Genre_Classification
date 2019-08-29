from yacs.config import CfgNode as CN

config = CN()
config.sep_only = True
config.pitch_only = False
config.sep = True
config.cuda = True
config.seed = 1984
config.mode = "train"
config.agent = CN()
config.agent.name = "FmaAgent"
config.max_epoch = 1024
config.non_blocking = True
config.distributed = True
config.half = False
config.lr = 0.0001
config.weight_decay = 0.
#config.opt = "radam"
config.opt = "sgd"
config.checkpoint_dir = "experiments/fma_debug"
#config.load_from = "checkpoint_17.pth.tar"
config.load_from = "checkpoint_15.pth.tar"
config.loss = "cross"

# Sep Data
config.sep_data = CN()
config.sep_data.data_loader_workers = 32
config.sep_data.batch_size = 11
config.sep_data.pin_memory = True
config.sep_data.distributed = config.distributed
trc = CN()
trc.data_root = "/root/shome/data_genre"
trc.mode = "training"
trc.dump_path = "/root/train_feat.h5"
trc.need_serialize = True
trc.serialized = True
trc.batch_size = config.sep_data.batch_size

tec = trc.clone()
tec.mode = "validation"
tec.dump_path = "/root/val_feat.h5"
tec.need_serialize = True
tec.serialized = True
tec.batch_size = int(trc.batch_size * 1.5)

config.sep_data.train = trc
config.sep_data.test = tec

config.model = CN()
config.model.name = 'VGGish'



def get_config():
    return config.clone()


