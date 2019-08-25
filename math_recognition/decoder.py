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

from allennlp.modules.token_embedders import Embedding

from math_recognition.attention import CaptioningAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CaptioningDecoder(nn.Module, Registrable):
    def __init__(self, vocab: Vocabulary):
        super(CaptioningDecoder, self).__init__()
        
        self.vocab = vocab
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, predicted_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
        
    def get_output_dim(self) -> int:
        raise NotImplementedError()

    # Input dim is dim of h and c
    def get_input_dim(self) -> int:
        raise NotImplementedError()

@CaptioningDecoder.register('image-captioning')
class ImageCaptioningDecoder(CaptioningDecoder):
    def __init__(self, vocab: Vocabulary, attention: CaptioningAttention, embedding_dim:int = 256, decoder_dim:int = 256):
        super(ImageCaptioningDecoder, self).__init__(vocab=vocab)
        
        self._vocab_size = self.vocab.get_vocab_size()
        self._embedding_dim = embedding_dim
        self._decoder_dim = decoder_dim

        self._embedding = Embedding(self._vocab_size, self._embedding_dim)
        self._attention = attention
        self._decoder_cell = nn.LSTMCell(self._embedding.get_output_dim() + self._attention.get_output_dim(), self._decoder_dim)
        self._linear = nn.Linear(self._decoder_dim, self._vocab_size)

    @overrides
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, predicted_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shape: (batch_size, embedding_dim)
        embedding = self._embedding(predicted_indices).float().view(-1, self._embedding_dim)
        
        # Shape: (batch_size, encoder_dim) (batch_size, h * w, 1)
        attention, attention_weights = self._attention(x, h)

        ## Change to not use teacher forcing all the time
        # Shape: (batch_size, decoder_dim) (batch_size, decoder_dim)
        h, c = self._decoder_cell(torch.cat([attention, embedding], dim=1), (h, c))
        
        # Get output predictions (one per character in vocab)
        # Shape: (batch_size, vocab_size)
        preds = self._linear(h)

        return h, c, preds, attention_weights
    
    @overrides
    def get_output_dim(self) -> int:
        return self._vocab_size
    
    @overrides
    def get_input_dim(self) -> int:
        return self._decoder_dim

@CaptioningDecoder.register('WAP')
class WAPDecoder(CaptioningDecoder):
    def __init__(self, vocab: Vocabulary, attention: CaptioningAttention, embedding_dim:int = 256, decoder_dim:int = 256):
        super(WAPDecoder, self).__init__(vocab=vocab)
        
        self._vocab_size = self.vocab.get_vocab_size()
        self._embedding_dim = embedding_dim
        self._decoder_dim = decoder_dim

        self._embedding = Embedding(self._vocab_size, self._embedding_dim)
        self._attention = attention
        self._decoder_cell = nn.GRUCell(self._embedding.get_output_dim() + self._attention.get_output_dim(), self._decoder_dim)
        self._linear = nn.Linear(self._decoder_dim, self._vocab_size)

    @overrides
    def forward(self, x: torch.Tensor, h: torch.Tensor, predicted_indices: torch.Tensor, sum_attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shape: (batch_size, embedding_dim)
        embedding = self._embedding(predicted_indices).float().view(-1, self._embedding_dim)
        
        # Shape: (batch_size, encoder_dim) (batch_size, h * w, 1) (batch_size, h * w)
        attention, attention_weights, sum_attention_weights = self._attention(x, h, sum_attention_weights)

        ## Change to not use teacher forcing all the time
        # Shape: (batch_size, decoder_dim) (batch_size, decoder_dim)
        h = self._decoder_cell(torch.cat([attention, embedding], dim=1), h)
        
        # Get output predictions (one per character in vocab)
        # Shape: (batch_size, vocab_size)
        preds = self._linear(h)

        return h, preds, attention_weights, sum_attention_weights
    
    @overrides
    def get_output_dim(self) -> int:
        return self._vocab_size
    
    @overrides
    def get_input_dim(self) -> int:
        return self._decoder_dim

@CaptioningDecoder.register('multiscale')
class MultiscaleDecoder(CaptioningDecoder):
    def __init__(self, vocab: Vocabulary, attention: CaptioningAttention, embedding_dim: int = 256, decoder_dim:int = 256):
        super(MultiscaleDecoder, self).__init__(vocab=vocab)

        self._vocab_size = self.vocab.get_vocab_size()
        self._embedding_dim = embedding_dim
        self._decoder_dim = decoder_dim
                
        self._embedding = Embedding(self._vocab_size, self._embedding_dim)
        self._dropout = nn.Dropout(0.1)
        # Output size of state cell must be decoder dim since state is transformed by the state cell
        self._state_cell = nn.GRUCell(self._embedding.get_output_dim(), self._decoder_dim)

        self._attention = attention
        self._decoder_cell = nn.GRUCell(self._attention.get_output_dim(), self._decoder_dim)

        self._linear = nn.Linear(self._decoder_dim, self._vocab_size)

    @overrides
    def forward(self, x: torch.Tensor, h: torch.Tensor, predicted_indices: torch.Tensor, sum_attention_weights_0: torch.Tensor, sum_attention_weights_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shape: (batch_size, embedding_dim)
        embedding = self._embedding(predicted_indices).float().view(-1, self._embedding_dim)
        embedding = self._dropout(embedding)
        
        # Shape: (batch_size, decoder_dim)
        h = self._state_cell(embedding, h)

        # Shape: (batch_size, encoder_dim) (batch_size, h * w, 1)
        attention, attention_weights, sum_attention_weights_0, sum_attention_weights_1 = self._attention(x, h, sum_attention_weights_0, sum_attention_weights_1)

        ## Change to not use teacher forcing all the time
        # Shape: (batch_size, decoder_dim) (batch_size, decoder_dim)
        h = self._decoder_cell(attention, h)

        # Get output predictions (one per character in vocab)
        # Shape: (batch_size, vocab_size)
        preds = self._linear(h)

        return h, preds, attention_weights, sum_attention_weights_0, sum_attention_weights_1
    
    @overrides
    def get_output_dim(self) -> int:
        return self._vocab_size
    
    @overrides
    def get_input_dim(self) -> int:
        return self._decoder_dim