'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.nnet.etc.pooling import StatsPooling
from core.nnet.etc.etc import ArcMarginProduct, AMLinear
from core.nnet.models.wav_encoder import WavEncoder
from core.nnet.models.transforms import SpectrumAug, Fbank


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, activation='relu'):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(4*growth_rate)
        self.conv2 = nn.Conv1d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'prelu':
            self.activation = torch.nn.PReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, activation='relu'):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(in_planes)
        # self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=2, stride=2, bias=False)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'prelu':
            self.activation = torch.nn.PReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.conv(self.activation(self.bn(x)))
        # out = F.avg_pool1d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block,
                 num_blocks,
                 n_classes,
                 fbank_config=None,
                 specaug_config=None,
                 activation='relu',
                 loss='amsoftmax',
                 m=0.35,
                 use_wav=False,
                 two_layer_fc=False,
                 mfcc_dim=41, embedding_size=256,
                 growth_rate=40, reduction=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        trans = []
        if fbank_config:
            trans.append(Fbank(**fbank_config))
            mfcc_dim = fbank_config['n_mels']
        if specaug_config:
            trans.append(SpectrumAug(**specaug_config))
        if trans:
            self.trans = nn.Sequential(*trans)
        else:
            self.trans = None

        num_planes = 2*growth_rate
        if use_wav:
            self.conv1 = WavEncoder(num_planes)
        else:
            self.conv1 = nn.Conv1d(mfcc_dim, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, num_blocks[0], activation=activation)
        num_planes += num_blocks[0] * growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, activation=activation)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, num_blocks[1], activation=activation)
        num_planes += num_blocks[1] * growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, activation=activation)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, num_blocks[2], activation=activation)
        num_planes += num_blocks[2] * growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, activation=activation)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, num_blocks[3], activation=activation)
        num_planes += num_blocks[3] * growth_rate

        self.bn = nn.BatchNorm1d(num_planes)


        # self.linear = nn.Linear(num_planes, num_classes)

        if not two_layer_fc:
            self.fc = nn.Linear(num_planes*2, embedding_size)
        else:
            self.fc = nn.Sequential(
                nn.Linear(num_planes*2, 512),
                nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                nn.Linear(512, embedding_size)
            )
        self.stats_pooling = StatsPooling()
        if loss == 'amsoftmax':
            print('using amsoftmax')
            self.cls_layer = AMLinear(embedding_size, n_classes, m=m)
        elif loss == 'arcface':
            print('using arcface')
            self.cls_layer = ArcMarginProduct(embedding_size, n_classes, m=m)
        elif loss == 'softmax':
            self.cls_layer = nn.Linear(embedding_size, n_classes)
        else:
            raise NotImplementedError

    def _make_dense_layers(self, block, in_planes, nblock, activation):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, activation=activation))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        if self.trans:
            x = self.trans(x)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = F.relu(self.bn(self.dense4(out)))

        out = self.stats_pooling(out)
        out = self.fc(out)
        if y is not None:
            out = self.cls_layer(out, y)
        else:
            out = self.cls_layer(out)
        # out = self.cls_layer(out, y)
        return out

    def extract(self, x):
        if self.trans:
            x = self.trans(x)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = F.relu(self.bn(self.dense4(out)))

        out = self.stats_pooling(out)
        out = self.fc(out)
        return out

class Deploy(DenseNet):
    def forward(self, x):
        x = x.transpose(1, 0)[None, :, :]
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = F.relu(self.bn(self.dense4(out)))

        out = self.stats_pooling(out)
        out = self.fc(out)
        return out

def DenseNet121(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 24, 16], **kwargs)

def DenseNet169(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 32, 32], **kwargs)

def DenseNet201Deploy(**kwargs):
    return Deploy(Bottleneck, [6, 12, 48, 32], **kwargs)

def DenseNet201(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 48, 32], **kwargs)

def DenseNet161(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 36, 24], **kwargs)
#
def DenseNet80(**kwargs):
    return DenseNet(Bottleneck, [6, 10, 14, 10], **kwargs)

# def densenet_80(**kwargs):
#     return DenseNet(Bottleneck, **kwargs)
# def test():
#     net = densenet_cifar()
#     x = torch.randn(1,3,32,32)
#     y = net(x)
#     print(y)

# test()

if __name__ == '__main__':
    model = DenseNet80(n_classes=1000, loss='amsoftmax', m=0.35, embedding_size=256)
    print(model)