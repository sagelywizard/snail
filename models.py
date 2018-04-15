"""A module containing the models described in the SNAIL paper
"""
import math

import torch.nn as nn
import snail


class OmniglotEmbedding(nn.Module):
    """A CNN which transforms a 1x28x28 image to a 64-dimensional vector
    """
    def __init__(self):
        super(OmniglotEmbedding, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.fc = nn.Linear(256, 64)

    def forward(self, minibatch):
        out = self.cnn1(minibatch)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        return self.fc(out.view(minibatch.size(0), -1))


class OmniglotModel(nn.Module):
    """
    Arguments:
        N (int): number of classes
        K (int): k-shot. i.e. number of examples
    """
    def __init__(self, N, K):
        super(OmniglotModel, self).__init__()
        T = N * K + 1
        layer_count = math.ceil(math.log(T)/math.log(2))
        self.mod0 = snail.AttentionBlock(65, 64, 32)
        self.mod1 = snail.TCBlock(65+32, T, 128)
        self.mod2 = snail.AttentionBlock(65+32+128*layer_count, 256, 128)
        self.mod3 = snail.TCBlock(65+32+128*layer_count+128, T, 128)
        self.mod4 = snail.AttentionBlock(65+32+2*128*layer_count+128, 512, 256)
        self.out_layer = nn.Conv1d(65+32+2*128*layer_count+128+256, N, 1)

    def forward(self, minibatch):
        out = self.mod0(minibatch)
        out = self.mod1(out)
        out = self.mod2(out)
        out = self.mod3(out)
        out = self.mod4(out)
        return self.out_layer(out)
