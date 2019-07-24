#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1h
from torch import nn
import torch


class Highway(nn.Module):
    def __init__(self, embed_size):

        super(Highway, self).__init__()

        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)

    def forward(self, X_conv_out):
        ''' (batch_size, n_features * embed_size) '''
        # # h_relu = self.projection(X_conv_out).clamp(min=0)
        # h_relu = nn.functional.relu(self.projection(X_conv_out))
        # y_pred = self.gate(h_relu)
        # return y_pred

        X_projection = nn.functional.relu(self.projection(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))
        # X_gate = self.gate(X_conv_out).clamp(min=0)
        X_highway = X_projection * X_gate + (1 - X_gate) * X_conv_out

        return X_highway

        # END YOUR CODE
