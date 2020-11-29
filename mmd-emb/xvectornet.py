import torch.nn as nn
import torch

class XvectorNet(nn.Module):
    def __init__(self, n_classes, mfcc_dim, embedding_layer='last'):
        super().__init__()
        self.frame_layers = nn.Sequential(
            cnn_bn_relu(mfcc_dim, 512, 5, dilation=1),
            cnn_bn_relu(512, 512, 3, dilation=2),
            cnn_bn_relu(512, 512, 3, dilation=3),
            cnn_bn_relu(512, 512, 1, dilation=1),
            cnn_bn_relu(512, 1500, 1, dilation=1),
        )
        self.utt_layers = nn.Sequential(
            nn.Linear(3000, 512),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.am_linear = nn.Linear(128, n_classes)
        self.stat_pooling = StatsPooling()

    def forward(self, x, y):
        x = self.frame_layers(x)
        x = self.stat_pooling(x)
        embedding = self.utt_layers(x)
        return self.am_linear(embedding)

    def extract(self, x):
        x = self.frame_layers(x)
        x = self.stat_pooling(x)
        embedding = self.utt_layers(x)
        return embedding


def cnn_bn_relu(indim, outdim, kernel_size, stride=1, dilation=1, padding=0):
    return nn.Sequential(
            nn.Conv1d(indim, outdim, kernel_size, stride=stride, dilation=dilation, padding=padding),
            nn.BatchNorm1d(outdim),
            torch.nn.ReLU(),
        )


class StatsPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        var = torch.sqrt((x - mean).pow(2).mean(-1) + 1e-5)
        return torch.cat([mean.squeeze(-1), var], -1)


