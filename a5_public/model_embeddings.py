#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        # A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        # End A4 code

        # YOUR CODE HERE for part 1j
        self.e_char = 50
        max_word = 21
        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(vocab.char2id),
            embedding_dim=self.e_char,
            padding_idx=vocab.char2id['<pad>'])

        self.embed_size = embed_size
        print(embed_size)
        print(self.embeddings)

        self.conv = CNN(e_char=self.e_char,
                        max_word=max_word,
                        output_shape=embed_size
                        )
        self.highway = Highway(embed_size)

        self.dropout = nn.Dropout(p=0.3)
        # END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sent_len, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sent_len, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        # A4 code
        # output = self.embeddings(input)
        # return output
        # End A4 code

        # YOUR CODE HERE for part 1j
        # print(input.shape)

        em = self.embeddings(input)

        sent_len, batch_size, max_word, max_char = em.shape

        reshape_dim = (sent_len * batch_size, max_word, self.e_char)
        # print(em.shape, reshape_dim)

        # thanks Erick!
        em = em.view(reshape_dim).transpose(1, 2)
        
        # print(em.shape)

        conv = self.conv(em)
        # print(conv.shape)
        highway = self.highway(conv)
        result = self.dropout(highway)
        # print(result.shape)
        
        result = result.view(sent_len, batch_size, self.embed_size)
        
        return result

        # END YOUR CODE
