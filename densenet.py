'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from core.nnet.etc.pooling import StatsPooling
# from core.nnet.etc.etc import ArcMarginProduct, AMLinear
# from core.nnet.models.wav_encoder import WavEncoder
# from core.nnet.models.transforms import SpectrumAug, Fbank

class AMLinear(nn.Module):
    def __init__(self, in_features, n_cls, m=0.35, s=30):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, n_cls))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.n_cls = n_cls
        self.s = s

    def forward(self, x, labels):
        w = self.weight
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        x = F.normalize(x, dim=1)

        cos_theta = torch.mm(x, ww)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        phi = cos_theta - self.m
        labels_one_hot = torch.zeros(len(labels), self.n_cls, device=labels.get_device()).scatter_(1, labels.unsqueeze(1), 1.)

        adjust_theta = self.s * torch.where(torch.eq(labels_one_hot, 1), phi, cos_theta)
        return adjust_theta, cos_theta



class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, m=0.50, s=30.0, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight.transpose(1, 0)))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=label.get_device())
        # one_hot = torch.zeros(cosine.size())

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output, phi



class StatsPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        var = torch.sqrt((x - mean).pow(2).mean(-1) + 1e-5)
        return torch.cat([mean.squeeze(-1), var], -1)


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
                 activation='relu',
                 loss='amsoftmax',
                 m=0.35,
                 two_layer_fc=False,
                 mfcc_dim=41, embedding_size=256,
                 growth_rate=40, reduction=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
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
            self.utt_layers = nn.Linear(num_planes * 2, embedding_size)
        else:
            self.utt_layers = nn.Sequential(
                nn.Linear(num_planes*2, 512),
                nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                nn.Linear(512, embedding_size)
            )
        self.stat_pooling = StatsPooling()
        if loss == 'amsoftmax':
            print('using amsoftmax')
            self.am_linear = AMLinear(embedding_size, n_classes, m=m)
        elif loss == 'arcface':
            print('using arcface')
            self.am_linear = ArcMarginProduct(embedding_size, n_classes, m=m)
        elif loss == 'softmax':
            self.am_linear = nn.Linear(embedding_size, n_classes)
        else:
            raise NotImplementedError

    def _make_dense_layers(self, block, in_planes, nblock, activation):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, activation=activation))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def frame_layers(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = F.relu(self.bn(self.dense4(out)))
        return out

    def forward(self, x, y=None):
        out = self.frame_layers(x)
        out = self.stat_pooling(out)
        out = self.utt_layers(out)
        if y is not None:
            out = self.am_linear(out, y)
        else:
            out = self.am_linear(out)
        # out = self.cls_layer(out, y)
        return out

    def extract(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = F.relu(self.bn(self.dense4(out)))

        out = self.stat_pooling(out)
        out = self.utt_layers(out)
        return out

class Deploy(DenseNet):
    def forward(self, x):
        x = x.transpose(1, 0)[None, :, :]
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = F.relu(self.bn(self.dense4(out)))

        out = self.stat_pooling(out)
        out = self.utt_layers(out)
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
