import numpy as np

from tqdm import tqdm
import shutil

import torch
from torch.backends import cudnn
from torch.autograd import Variable

from graphs.models.conv_tasnet import ConvTasNet
from datasets.musdb18 import Musdb18DB

from torch.optim import lr_scheduler

from utils.metrics import AverageMeter, IOUMetric
from utils.misc import print_cuda_statistics

from agents.base import BaseAgent

cudnn.benchmark = True


class TasNetAgent(BaseAgent):
    """
    This class will be responsible for handling the whole process of our architecture.
    """

    def __init__(self, config):
        super().__init__(config)
        # Create an instance from the Model

        # define model
        self.model = ConvTasNet(self.config.model)
        # Create an instance from the data loader
        self.data_loader = Musdb18DB(self.config.data)
        self.cuda = config.cuda

        if self.cuda:
            torch.cuda.manual_seed_all(self.config.seed)
            self.device = torch.device("cuda")
            #torch.cuda.set_device(self.config.gpu_device)
            #torch.cuda.set_device(4)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()

        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")

        self.model = self.model.to(self.device)

        self.current_epoch = 0
        self.current_iteration = 0


    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        assert self.config.mode in ['train', 'test', 'random']
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """

        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()


    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))

        # Set the model to be in training mode (for batchnorm)
        self.model.train()

        for x, y in tqdm_batch:
            ## TODO
            x, y = x.float(), y.float()
            if self.cuda:
                x, y = x.pin_memory().cuda(non_blocking=self.config.non_blocking), y.cuda(non_blocking=self.config.non_blocking)

            __import__('pdb').set_trace() 
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            epoch_loss.update(cur_loss.item())
            _, pred_max = torch.max(pred, 1)
            metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

            self.current_iteration += 1
            # exit(0)

        epoch_acc, _, epoch_iou_class, epoch_mean_iou, _ = metrics.evaluate()
        self.summary_writer.add_scalar("epoch-training/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/mean_iou", epoch_mean_iou, self.current_iteration)
        tqdm_batch.close()

        print("Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + " - acc-: " + str(
            epoch_acc) + "- mean_iou: " + str(epoch_mean_iou) + "\n iou per class: \n" + str(
            epoch_iou_class))

    def validate(self):
        """
        One epoch validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        metrics = IOUMetric(self.config.num_classes)

        for x, y in tqdm_batch:
            if self.cuda:
                assert False
                #x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
            x, y = Variable(x), Variable(y)
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during Validation.')

            _, pred_max = torch.max(pred, 1)
            metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

            epoch_loss.update(cur_loss.item())

        epoch_acc, _, epoch_iou_class, epoch_mean_iou, _ = metrics.evaluate()
        self.summary_writer.add_scalar("epoch_validation/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/mean_iou", epoch_mean_iou, self.current_iteration)

        print("Validation Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + " - acc-: " + str(
            epoch_acc) + "- mean_iou: " + str(epoch_mean_iou) + "\n iou per class: \n" + str(
            epoch_iou_class))

        tqdm_batch.close()

        return epoch_mean_iou, epoch_loss.val

    def test(self):
        # TODO
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
