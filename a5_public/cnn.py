#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1i
from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self, e_char, max_word, output_shape, conv_dim=None, k=5):

        super(CNN, self).__init__()

        self.conv = torch.nn.Conv1d(e_char, output_shape, kernel_size=k, bias=True)
        self.maxpool=nn.MaxPool1d(kernel_size=max_word - k + 1)

    def forward(self, reshaped):
        ''' (batch_size, n_features * embed_size) '''

        conv=self.conv(reshaped)
        conv_r=nn.functional.relu(conv)
        maxpooled=self.maxpool(conv_r).squeeze()

        return maxpooled

        # END YOUR CODE

# END YOUR CODE
