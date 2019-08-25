import os
import random
from typing import Dict, Tuple
from overrides import overrides

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import allennlp

from allennlp.common import Registrable, Params
from allennlp.data.vocabulary import Vocabulary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CaptioningAttention(nn.Module, Registrable):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_output_dim(self) -> int:
        raise NotImplementedError()

@CaptioningAttention.register('image-captioning')
class ImageCaptioningAttention(CaptioningAttention):
    def __init__(self, encoder_dim: int = 512, decoder_dim: int = 256, attention_dim: int = 256, doubly_stochastic_attention: bool = True) -> None:
        super().__init__()
                
        self._encoder_dim = encoder_dim
        self._decoder_dim = decoder_dim
        self._attention_dim = attention_dim
        
        self._doubly_stochastic_attention = doubly_stochastic_attention
        
        self._encoder_attention = nn.Linear(self._encoder_dim, self._attention_dim)
        self._decoder_attention = nn.Linear(self._decoder_dim, self._attention_dim)
        self._attention = nn.Linear(self._attention_dim, 1)

        if self._doubly_stochastic_attention:
            self._f_beta = nn.Linear(self._decoder_dim, self._encoder_dim)

    @overrides
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: (batch_size, height * width, attention_dim)
        encoder_attention = self._encoder_attention(x)
        # Shape: (batch_size, 1, attention_dim)
        decoder_attention = self._decoder_attention(h).unsqueeze(1)

        # Shape: (batch_size, height * width)
        # Can't concat attention since encoder returns h*w and decoder returns 1
        attention = self._attention(torch.tanh(encoder_attention + decoder_attention)).squeeze(2)

        # No need for masked softmax since all encoder pixels are available and hidden state of rnn isn't masked
        # Shape: (batch_size, h * w, 1)
        attention_weights = torch.softmax(attention, dim=1).unsqueeze(2)

        # Shape: (batch_size, encoder_dim)
        attention = (x * attention_weights).sum(dim=1)
        
        if self._doubly_stochastic_attention:     
            # Shape: (batch_size, encoder_dim)
            gate = torch.sigmoid(self._f_beta(h))
            # Shape: (batch_size, encoder_dim)
            attention = gate * attention
        
        return attention, attention_weights
    
    @overrides
    def get_output_dim(self) -> int:
        return self._encoder_dim

@CaptioningAttention.register('WAP')
class WAPAttention(CaptioningAttention):
    def __init__(self, encoder_dim: int = 512, decoder_dim: int = 256, attention_dim: int = 256, kernel_size: int = 5, padding: int=2) -> None:
        super().__init__()
                
        self._encoder_dim = encoder_dim
        self._decoder_dim = decoder_dim
        self._attention_dim = attention_dim
        self._kernel_size = kernel_size
        
        self._encoder_attention = nn.Linear(self._encoder_dim, self._attention_dim)
        self._decoder_attention = nn.Linear(self._decoder_dim, self._attention_dim)
        
        # If kernel size is changed, padding needs to also change
        # Not sure if original uses padding; needed here since need same dimension inputs to attention
        self._coverage = nn.Conv2d(1, self._attention_dim, kernel_size, padding=padding)
        self._coverage_attention = nn.Linear(self._attention_dim, self._attention_dim)
        
        self._attention = nn.Linear(self._attention_dim, 1)

    @overrides
    def forward(self, x: torch.Tensor, h: torch.Tensor, sum_attention_weights: torch.Tensor, height: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: (batch_size, height * width, attention_dim)
        encoder_attention = self._encoder_attention(x)

        # Shape: (batch_size, 1, attention_dim)
        decoder_attention = self._decoder_attention(h).unsqueeze(1)

        # Get coverage over sum correctly when batch size at timestep isn't local batch size 
        # Need to clone sum_attention_weights since it's modified by an in-place operation 
        # Assumes 4:1 aspect ratio
        # Shape: (batch_size, height * width, attention_dim)
        # Shape: (batch_size, height * width, attention_dim)
        coverage = self._coverage(sum_attention_weights[:encoder_attention.shape[0]].view(-1,1, height, height * 4).clone()).view(-1, height * height * 4, self._attention_dim)
        coverage_attention = self._coverage_attention(coverage)

        # Shape: (batch_size, height * width)
        attention = self._attention(torch.tanh(encoder_attention + decoder_attention + coverage_attention)).squeeze(2)

        # No need for masked softmax since all encoder pixels are available and hidden state of rnn isn't masked
        # Shape: (batch_size, h * w, 1)
        attention_weights = torch.softmax(attention, dim=1).unsqueeze(2)

        # Update sum correctly when batch size at timestep isn't local batch size 
        # Shape: (batch_size, h * w)
        sum_attention_weights[:attention_weights.shape[0]] += attention_weights.view(-1, attention_weights.shape[1])

        # Shape: (batch_size, encoder_dim)
        attention = (x * attention_weights).sum(dim=1)
        
        return attention, attention_weights, sum_attention_weights
    
    @overrides
    def get_output_dim(self) -> int:
        return self._encoder_dim

@CaptioningAttention.register('multiscale')
class MultiscaleAttention(CaptioningAttention):
    def __init__(self, main_attention: CaptioningAttention, multiscale_attention: CaptioningAttention, height_1: int = 4, height_2: int = 8) -> None:
        super().__init__()

        self._main_attention = main_attention
        self._multiscale_attention = multiscale_attention
        
        self._height_1 = height_1
        self._height_2 = height_2

    @overrides
    def forward(self, x: torch.Tensor, h: torch.Tensor, sum_attention_weights_0: torch.Tensor, sum_attention_weights_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        main_features, multiscale_features = x[0], x[1]
        
        main_attention, main_attention_weights, sum_attention_weights_0 = self._main_attention(main_features, h, sum_attention_weights_0, height=self._height_1)
        multiscale_attention, multiscale_attention_weights, sum_attention_weights_1 = self._multiscale_attention(multiscale_features, h, sum_attention_weights_1, height=self._height_2)
        
        attention = torch.cat([main_attention, multiscale_attention], dim=1)
        
        return attention, (main_attention_weights, multiscale_attention_weights), sum_attention_weights_0, sum_attention_weights_1
    
    @overrides
    def get_output_dim(self) -> int:
        return self._main_attention._encoder_dim + self._multiscale_attention._encoder_dim
