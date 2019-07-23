#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1i
from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self, reshape_dim, conv_dim=None, k=5):

        super(CNN, self).__init__()

        self.conv = torch.nn.Conv1d(reshape_dim, conv_dim, kernel_size=k)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)

    def forward(self, reshaped):
        ''' (batch_size, n_features * embed_size) '''

        conv = self.conv(reshaped).clamp(min=0)
        maxpooled = self.maxpool(conv)
        sq = torch.squeeze(maxpooled, -1)
        print(maxpooled.shape, sq.shape)
        return sq

        # END YOUR CODE

# END YOUR CODE
