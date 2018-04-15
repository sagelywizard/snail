#!/usr/bin/python3
"""A PyTorch implementation of the SNAIL building blocks.

This module implements the three blocks in the _A Simple Neural Attentive
Meta-Learner_ paper Mishra et al.

    URL: https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW

The three building blocks are the following:
    - A dense block, built with causal convolutions.
    - A TC Block, built with a stack of dense blocks.
    - An attention block, similar to the attention mechanism described by
      Vaswani et al (2017).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CausalConv1d(nn.Module):
    """A 1D causal convolution layer.

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions per step, and T is the number of steps.
    Output: (B, D_out, T), where B is the minibatch size, D_out is the number
        of dimensions in the output, and T is the number of steps.

    Arguments:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            padding = self.padding,
            dilation = dilation
        )

    def forward(self, minibatch):
        return self.causal_conv(minibatch)[:, :, :-self.padding]


class DenseBlock(nn.Module):
    """Two parallel 1D causal convolution layers w/tanh and sigmoid activations

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.

    Arguments:
        in_channels (int): number of input channels
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, filters, dilation=1):
        super(DenseBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation
        )
        self.causal_conv2 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation
        )

    def forward(self, minibatch):
        tanh = F.tanh(self.causal_conv1(minibatch))
        sig = F.sigmoid(self.causal_conv2(minibatch))
        out = torch.cat([minibatch, tanh*sig], dim=1)
        return out


class TCBlock(nn.Module):
    """A stack of DenseBlocks which dilates to desired sequence length

    The TCBlock adds `ceil(log_2(seq_len))*filters` channels to the output.

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.

    Arguments:
        in_channels (int): channels for the input
        seq_len (int): length of the sequence. The number of denseblock layers
            is log base 2 of `seq_len`.
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, seq_len, filters):
        super(TCBlock, self).__init__()
        layer_count = math.ceil(math.log(seq_len)/math.log(2))
        blocks = []
        channel_count = in_channels
        for layer in range(layer_count):
            block = DenseBlock(channel_count, filters, dilation=2**layer)
            blocks.append(block)
            channel_count += filters
        self.blocks = nn.Sequential(*blocks)

    def forward(self, minibatch):
        return self.blocks(minibatch)


class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)

    The input of the AttentionBlock is `BxDxT` where `B` is the input
    minibatch size, `D` is the dimensions of each feature, `T` is the length of
    the sequence.

    The output of the AttentionBlock is `Bx(D+V)xT` where `V` is the size of the
    attention values.

    Arguments:
        input_dims (int): the number of dimensions (or channels) of each element
            in the input sequence
        k_size (int): the size of the attention keys
        v_size (int): the size of the attention values
    """
    def __init__(self, input_dims, k_size, v_size):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(input_dims, k_size)
        self.query_layer = nn.Linear(input_dims, k_size)
        self.value_layer = nn.Linear(input_dims, v_size)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2,1))
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
        mask = torch.triu(mask, 1)
        mask = mask.unsqueeze(0).expand_as(logits)
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits / self.sqrt_k, dim=2)
        read = torch.bmm(probs, values)
        return torch.cat([minibatch, read], dim=2).permute(0,2,1)
