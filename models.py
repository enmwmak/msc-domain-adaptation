import torch.nn as nn
import torch
import torchaudio
import h5py
import numpy as np
import torch.nn.functional as F


class Xvector(nn.Module):
    def __init__(self, n_classes,):
        super().__init__()
        self.frame_layers = nn.Sequential(
            cnn_bn_relu(23, 512, 5, dilation=1),
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
        self.stats_pooling = StatsPooling()

    def forward(self, x, y):
        x = self.frame_layers(x)
        x = self.stats_pooling(x)
        embedding = self.utt_layers(x)
        return self.am_linear(embedding)

    def extract(self, x):
        x = self.frame_layers(x)
        x = self.stats_pooling(x)
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


if __name__ == '__main__':
    # for source
    frames_src = self.model.module.frame_layers(mfcc_src)  # 128X1500X400
    stats_src = self.model.module.stat_pooling(frames_src)
    embed_src = self.model.module.utt_layers(stats_src)
    logit_src, logit_nomargin_src = self.model.module.am_linear(embed_src, spk_ids_src)

    # for target
    frames_tgt = self.model.module.frame_layers(mfcc_tgt)
    stats_tgt = self.model.module.stat_pooling(frames_tgt)
    embed_tgt = self.model.module.utt_layers(stats_tgt)

    cls_loss = F.cross_entropy(logit_src, spk_ids_src)
    # domain_loss = self.domain_loss_weight * self.unsup_loss(embed_tgt, embed_src)
    # X 64X1500X400
    sample_idx = np.random.randint(stats_tgt.reshape[0] * stats_tgt.reshape[2] - 3000)
    domain_loss = self.domain_loss_weight * self.unsup_loss(embed_tgt, embed_src) \
                  + self.unsup_loss(frames_src.permute(0, 2, 1).reshape[-1, 1500][sample_idx:sample_idx + 3000],
                                    frames_src.permute(0, 2, 1).reshape[-1, 1500][sample_idx:sample_idx + 3000])
    loss = cls_loss + domain_loss
