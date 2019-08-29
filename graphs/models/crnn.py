"""
Mnist tutorial main model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ..weights_initializer import weights_init

class SequenceWise(nn.Module):
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        ori_shape = x.shape
        x = x.flatten(start_dim=0, end_dim=1)
        x = self.module(x)
        new_shape = ori_shape[:2] + x.shape[1:]
        x = x.reshape(new_shape)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

def init_weights(mod):
    for m in mod.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

class CRNN(nn.Module):
    def __init__(self, cfg):
        super(CRNN, self).__init__()

        self.module1 = SequenceWise(nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(1, 5), padding=(0, 2), stride=(1, 5)),
                nn.ReLU(),
                nn.BatchNorm2d(64)
                ))

        self.module2 = SequenceWise(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 5), padding=(1, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(64)
                ))

        self.module3 = SequenceWise(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(64)
                ))

        self.module4 = SequenceWise(nn.Sequential(
                nn.Conv2d(64, 16, kernel_size=(3, 15), padding=(1, 7)),
                nn.ReLU(),
                nn.BatchNorm2d(16)
                ))

        self.module5 = SequenceWise(nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=(1, 1)),
                nn.ReLU()
                ))
        self.rnn = torch.nn.GRU(input_size=61, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(256, 62)
        self.rnn.apply(init_weights)

    def forward(self, x):
        group_len = 50
        ori_shape = x.shape
        new_shape = [ori_shape[0], ori_shape[2] // 50, ori_shape[1], 50, ori_shape[3]]        
        x = x.reshape(new_shape)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.module5(x)
        x = x.squeeze(dim=2)
        x = x.flatten(start_dim=1, end_dim=2)
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    test_model = CRNN()
    test_input = torch.ones((3, 1, 500, 301))
    test_model(test_input)


