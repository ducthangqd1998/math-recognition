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
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.token_embedders import Embedding

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.beam_search import BeamSearch

from allennlp.training.metrics import F1Measure, BLEU

from math_recognition.metrics import Exprate
from math_recognition.encoder import Encoder
from math_recognition.decoder import CaptioningDecoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@Model.register('image-captioning')
class ImageCaptioning(Model):
    def __init__(self, vocab: Vocabulary, encoder: Encoder, decoder: CaptioningDecoder, max_timesteps: int = 75, teacher_forcing: bool = True, scheduled_sampling_ratio: float = 1, beam_size: int = 10) -> None:
        super().__init__(vocab)

        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)
        self._pad_index = self.vocab.get_token_index('@@PADDING@@')

        self._max_timesteps = max_timesteps
        self._teacher_forcing = teacher_forcing
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._beam_size = beam_size

        self._encoder = encoder
        self._decoder = decoder

        self._init_h = nn.Linear(self._encoder.get_output_dim(), self._decoder.get_input_dim())
        self._init_c = nn.Linear(self._encoder.get_output_dim(), self._decoder.get_input_dim())

        self.beam_search = BeamSearch(self._end_index, self._max_timesteps, self._beam_size)

        self._bleu = BLEU(exclude_indices={self._start_index, self._end_index, self._pad_index})
        self._exprate = Exprate(self._end_index, self.vocab)

        self._attention_weights = None
        
    def _init_hidden(self, encoder: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_encoder = encoder.mean(dim=1)
        
        # Shape: (batch_size, decoder_dim)
        initial_h = self._init_h(mean_encoder)
        # Shape: (batch_size, decoder_dim)
        initial_c = self._init_c(mean_encoder)

        return initial_h, initial_c
    
    def _decode(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get data from state
        metadata = state['metadata']
        x = state['x']
        h = state['h']
        c = state['c']
        label = state['label']
        mask = state['mask']
        
        # Get actual size of current batch
        local_batch_size = x.shape[0]

        # Sort data to be able to only compute relevent parts of the batch at each timestep
        # Shape: (batch_size)
        lengths = mask.sum(dim=1)
        # Shape: (batch_size) (batch_size)
        sorted_lengths, indices = lengths.sort(dim=0, descending=True)
        # Computing last timestep isn't necessary with labels since last timestep is eos token or pad token 
        timesteps = sorted_lengths[0] - 1

        # Shape: (batch_size, ?)
        # Shape: (batch_size, height * width, encoder_dim)
        # Shape: (batch_size, decoder_dim)
        # Shape: (batch_size, decoder_dim)
        # Shape: (batch_size, timesteps)
        # Shape: (batch_size, timesteps)
        metadata = [metadata[i] for i in indices]
        x = x[indices]
        h = h[indices]
        c = c[indices]
        label = label[indices]        
        mask = mask[indices]
        
        # Shape: (batch_size, 1)
        predicted_indices = torch.LongTensor([[self._start_index]] * local_batch_size).to(device).view(-1, 1)
        
        # Shape: (batch_size, timesteps, vocab_size)
        predictions = torch.zeros(local_batch_size, timesteps, self._decoder.get_output_dim(), device=device)
        attention_weights = torch.zeros(local_batch_size, timesteps, self._encoder.get_feature_map_size(), device=device)
        
        for t in range(timesteps):
            # Shape: (batch_offset)
            batch_offset = sum([l > t for l in sorted_lengths.tolist()])

            # Only compute data in valid timesteps
            # Shape: (batch_offset, height * width, encoder_dim)
            # Shape: (batch_offset, decoder_dim)
            # Shape: (batch_offset, decoder_dim)
            # Shape: (batch_offset, 1)
            x_t = x[:batch_offset]
            h_t = h[:batch_offset]
            c_t = c[:batch_offset]
            predicted_indices_t = predicted_indices[:batch_offset]
            
            # Decode timestep
            # Shape: (batch_size, decoder_dim) (batch_size, decoder_dim) (batch_size, vocab_size), (batch_size, encoder_dim, 1)
            h, c, preds, attention_weight = self._decoder(x_t, h_t, c_t, predicted_indices_t)
            
            # Get new predicted indices to pass into model at next timestep
            # Use teacher forcing if chosen
            if self._teacher_forcing:
                # Send next timestep's label to next timestep
                # Shape: (batch_size, 1)
                predicted_indices = label[:batch_offset, t + 1].view(-1, 1)
            else:
                # Shape: (batch_size, 1)
                predicted_indices = torch.argmax(preds, dim=1).view(-1, 1)
            
            # Save preds
            predictions[:batch_offset, t, :] = preds
            attention_weights[:batch_offset, t, :] = attention_weight.view(-1, self._encoder.get_feature_map_size())
            
        # Update state and add logits
        state['metadata'] = metadata
        state['x'] = x
        state['h'] = h
        state['c'] = c
        state['label'] = label
        state['mask'] = mask
        state['attention_weights'] = attention_weights
        state['logits'] = predictions
            
        return state
    
    def _beam_search_step(self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Group_size is batch_size * beam_size except for first decoding timestep where it is batch_size
        # Shape: (group_size, decoder_dim) (group_size, decoder_dim) (group_size, vocab_size)
        h, c, predictions, attention_weights = self._decoder(state['x'], state['h'], state['c'], last_predictions)
        
        if self._attention_weights is not None:
            attention_weights = attention_weights.view(-1, self._beam_size, 1, self._encoder.get_feature_map_size())
            self._attention_weights = torch.cat([self._attention_weights, attention_weights[:, 0, :, :]], dim=1)
        else:
            attention_weights = attention_weights.view(-1, 1, self._encoder.get_feature_map_size())
            self._attention_weights = attention_weights

        # Update state
        # Shape: (group_size, decoder_dim)
        state['h'] = h
        # Shape: (group_size, decoder_dim)
        state['c'] = c

        # Run log_softmax over logit predictions
        # Shape: (group_size, vocab_size)
        log_preds = F.log_softmax(predictions, dim=1)

        return log_preds, state
    
    def _beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get data from state
        x = state['x']
        h = state['h']
        c = state['c']
        
        # Get actual size of current batch
        local_batch_size = x.shape[0]

        # Beam search wants initial preds of shape: (batch_size)
        # Shape: (batch_size)
        initial_indices = torch.LongTensor([[self._start_index]] * local_batch_size).to(device).view(-1)
        
        state = {'x': x, 'h': h, 'c': c}
        
        # Timesteps returned aren't necessarily max_timesteps
        # Shape: (batch_size, beam_size, timesteps), (batch_size, beam_size)
        
        self._attention_weights = None
        
        predictions, log_probabilities = self.beam_search.search(initial_indices, state, self._beam_search_step)

        # Only keep best predictions from beam search
        # Shape: (batch_size, timesteps)
        predictions = predictions[:, 0, :].view(local_batch_size, -1)
        
        return predictions
        
    @overrides
    def forward(self, metadata: object, img: torch.Tensor, label: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Encode the image
        # Shape: (batch_size, height * width, encoder_dim)
        x = self._encoder(img)

        state = {'metadata': metadata, 'x': x}
        # Compute loss on train and val
        if label is not None:
            # Initialize h and c
            # Shape: (batch_size, decoder_dim)
            state['h'], state['c'] = self._init_hidden(x)

            # Convert label dict to tensor since label isn't an input to the model and get mask
            # Shape: (batch_size, timesteps)
            state['mask'] = get_text_field_mask(label).to(device)
            # Shape: (batch_size, timesteps)
            state['label'] = label['tokens']

            # Decode encoded image and get loss on train and val
            state = self._decode(state)

            # Loss shouldn't be computed on start token
            state['mask'] = state['mask'][:, 1:].contiguous()
            state['target'] = state['label'][:, 1:].contiguous()

            # Compute cross entropy loss
            state['loss'] = sequence_cross_entropy_with_logits(state['logits'], state['target'], state['mask'])
            # Doubly stochastic regularization
            state['loss'] += ((1 - torch.sum(state['attention_weights'], dim=1)) ** 2).mean()

        # Decode encoded image with beam search on val and test
        if not self.training:
            # (Re)initialize h and c
            state['h'], state['c'] = self._init_hidden(state['x'])
            
            # Run beam search
            state['out'] = self._beam_search(state)

            # Save attention weights
            state['attention_weights'] = self._attention_weights

            # Compute validation scores
            if 'label' in state:
                # doesn't work in aug 2019
                # self._bleu(state['out'].float(), state['target'].float())
                self._exprate(state['out'], state['target'])
            
        # Set out to logits while training
        else:
            state['out'] = state['logits']
            
        return state
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Return Bleu score if possible
        if not self.training:
            # doesn't work in aug 2019
            # metrics.update(self._bleu.get_metric(reset))
            metrics.update(self._exprate.get_metric(reset))
            
        return metrics
        
    def _trim_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        for b in range(predictions.shape[0]):
            # Shape: (timesteps)
            predicted_index = predictions[b]
            # Set last predicted index to eos token in case there are no predicted eos tokens
            predicted_index[-1] = self._end_index

            # Get index of first eos token
            # Shape: (timesteps)
            mask = predicted_index == self._end_index
            # Work around for pytorch not having an easy way to get the first non-zero index
            eos_token_idx = list(mask.cpu().numpy()).index(1)
            
            # Set prediction at eos token's timestep to eos token
            predictions[b, eos_token_idx] = self._end_index
            # Replace all timesteps after first eos token with pad token
            predictions[b, eos_token_idx + 1:] = self._pad_index

        return predictions

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Trim test preds to first eos token
        # Shape: (batch_size, timesteps)
        output_dict['out'] = self._trim_predictions(output_dict['out'])

        return output_dict

@Model.register('WAP')
class WAP(ImageCaptioning):
    def __init__(self, vocab: Vocabulary, encoder: Encoder, decoder: CaptioningDecoder, max_timesteps: int = 75, teacher_forcing: bool = True, scheduled_sampling_ratio: float = 1, beam_size: int = 10) -> None:
        super().__init__(vocab, encoder, decoder, max_timesteps, teacher_forcing, scheduled_sampling_ratio, beam_size)
        
    def _init_hidden(self, encoder: torch.Tensor) -> torch.Tensor:
        mean_encoder = encoder.mean(dim=1)
        
        # Shape: (batch_size, decoder_dim)
        initial_h = self._init_h(mean_encoder)

        return initial_h
    
    def _decode(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get data from state
        metadata = state['metadata']
        x = state['x']
        h = state['h']
        label = state['label']
        mask = state['mask']
        
        # Get actual size of current batch
        local_batch_size = x.shape[0]

        # Sort data to be able to only compute relevent parts of the batch at each timestep
        # Shape: (batch_size)
        lengths = mask.sum(dim=1)
        # Shape: (batch_size) (batch_size)
        sorted_lengths, indices = lengths.sort(dim=0, descending=True)
        # Computing last timestep isn't necessary with labels since last timestep is eos token or pad token 
        timesteps = sorted_lengths[0] - 1

        # Shape: (batch_size, ?)
        # Shape: (batch_size, height * width, encoder_dim)
        # Shape: (batch_size, decoder_dim)
        # Shape: (batch_size, timesteps)
        # Shape: (batch_size, timesteps)
        metadata = [metadata[i] for i in indices]
        x = x[indices]
        h = h[indices]
        label = label[indices]        
        mask = mask[indices]
        
        # Shape: (batch_size, 1)
        predicted_indices = torch.LongTensor([[self._start_index]] * local_batch_size).to(device).view(-1, 1)
        
        # Shape: (batch_size, timesteps, vocab_size)
        predictions = torch.zeros(local_batch_size, timesteps, self._decoder.get_output_dim(), device=device)
        attention_weights = torch.zeros(local_batch_size, timesteps, self._encoder.get_feature_map_size(), device=device)
        sum_attention_weights = torch.zeros(local_batch_size, self._encoder.get_feature_map_size(), device=device)

        for t in range(timesteps):
            # Shape: (batch_offset)
            batch_offset = sum([l > t for l in sorted_lengths.tolist()])

            # Only compute data in valid timesteps
            # Shape: (batch_offset, height * width, encoder_dim)
            # Shape: (batch_offset, decoder_dim)
            # Shape: (batch_offset, 1)
            x_t = x[:batch_offset]
            h_t = h[:batch_offset]
            predicted_indices_t = predicted_indices[:batch_offset]
            
            # Decode timestep
            # Shape: (batch_size, decoder_dim) (batch_size, vocab_size), (batch_size, encoder_dim, 1), (batch_size, height * width)
            h, preds, attention_weight, sum_attention_weights = self._decoder(x_t, h_t, predicted_indices_t, sum_attention_weights)
            
            # Get new predicted indices to pass into model at next timestep
            # Use teacher forcing if chosen
            if self._teacher_forcing:
                # Send next timestep's label to next timestep
                # Shape: (batch_size, 1)
                predicted_indices = label[:batch_offset, t + 1].view(-1, 1)
            else:
                # Shape: (batch_size, 1)
                predicted_indices = torch.argmax(preds, dim=1).view(-1, 1)
            
            # Save preds
            predictions[:batch_offset, t, :] = preds
            attention_weights[:batch_offset, t, :] = attention_weight.view(-1, self._encoder.get_feature_map_size())
            
        # Update state and add logits
        state['metadata'] = metadata
        state['x'] = x
        state['h'] = h
        state['label'] = label
        state['mask'] = mask
        state['attention_weights'] = attention_weights
        state['logits'] = predictions
            
        return state
    
    def _beam_search_step(self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Group_size is batch_size * beam_size except for first decoding timestep where it is batch_size
        # Shape: (group_size, decoder_dim) (group_size, vocab_size) (?) (group_size, height * width)
        h, predictions, attention_weights, sum_attention_weights = self._decoder(state['x'], state['h'], last_predictions, state['sum_attention_weights'])
        
        if self._attention_weights is not None:
            attention_weights = attention_weights.view(-1, self._beam_size, 1, self._encoder.get_feature_map_size())
            self._attention_weights = torch.cat([self._attention_weights, attention_weights[:, 0, :, :]], dim=1)
        else:
            attention_weights = attention_weights.view(-1, 1, self._encoder.get_feature_map_size())
            self._attention_weights = attention_weights

        # Update state
        # Shape: (group_size, decoder_dim)
        # Shape: (group_size, height * width)
        state['h'] = h
        state['sum_attention_weights'] = sum_attention_weights

        # Run log_softmax over logit predictions
        # Shape: (group_size, vocab_size)
        log_preds = F.log_softmax(predictions, dim=1)

        return log_preds, state
    
    def _beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get data from state
        x = state['x']
        h = state['h']
        
        # Get actual size of current batch
        local_batch_size = x.shape[0]

        # Beam search wants initial preds of shape: (batch_size)
        # Shape: (batch_size)
        # Shape: (batch_size, height * width)    
        initial_indices = torch.LongTensor([[self._start_index]] * local_batch_size).to(device).view(-1)
        sum_attention_weights = torch.zeros(local_batch_size, self._encoder.get_feature_map_size(), device=device)

        state = {'x': x, 'h': h, 'sum_attention_weights': sum_attention_weights}
        
        self._attention_weights = None

        # Timesteps returned aren't necessarily max_timesteps
        # Shape: (batch_size, beam_size, timesteps), (batch_size, beam_size)        
        predictions, log_probabilities = self.beam_search.search(initial_indices, state, self._beam_search_step)

        # Only keep best predictions from beam search
        # Shape: (batch_size, timesteps)
        predictions = predictions[:, 0, :].view(local_batch_size, -1)
        
        return predictions
        
    @overrides
    def forward(self, metadata: object, img: torch.Tensor, label: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Encode the image
        # Shape: (batch_size, height * width, encoder_dim)
        x = self._encoder(img)

        state = {'metadata': metadata, 'x': x}
        # Compute loss on train and val
        if label is not None:
            # Initialize h and c
            # Shape: (batch_size, decoder_dim)
            state['h'] = self._init_hidden(x)

            # Convert label dict to tensor since label isn't an input to the model and get mask
            # Shape: (batch_size, timesteps)
            state['mask'] = get_text_field_mask(label).to(device)
            # Shape: (batch_size, timesteps)
            state['label'] = label['tokens']

            # Decode encoded image and get loss on train and val
            state = self._decode(state)

            # Loss shouldn't be computed on start token
            state['mask'] = state['mask'][:, 1:].contiguous()
            state['target'] = state['label'][:, 1:].contiguous()

            # Compute cross entropy loss
            state['loss'] = sequence_cross_entropy_with_logits(state['logits'], state['target'], state['mask'])
            # No doubly stochastic loss in WAP
            # Doubly stochastic regularization
#             state['loss'] += ((1 - torch.sum(state['attention_weights'], dim=1)) ** 2).mean()

        # Decode encoded image with beam search on val and test
        if not self.training:
            # (Re)initialize h
            state['h'] = self._init_hidden(state['x'])
            
            # Run beam search
            state['out'] = self._beam_search(state)

            # Save attention weights
            state['attention_weights'] = self._attention_weights

            # Compute validation scores
            if 'label' in state:
                self._bleu(state['out'], state['target'])
                self._exprate(state['out'], state['target'])
            
        # Set out to logits while training
        else:
            state['out'] = state['logits']
            
        return state
    
@Model.register('multiscale')
class Multiscale(ImageCaptioning):
    def __init__(self, vocab: Vocabulary, encoder: Encoder, decoder: CaptioningDecoder, max_timesteps: int = 75, teacher_forcing: bool = True, scheduled_sampling_ratio: float = 1, beam_size: int = 10) -> None:
        super().__init__(vocab, encoder, decoder, max_timesteps, teacher_forcing, scheduled_sampling_ratio, beam_size)

    @overrides
    def _init_hidden(self, encoder: torch.Tensor) -> torch.Tensor:
        mean_encoder = encoder[0].mean(dim=1)
        
        # Shape: (batch_size, decoder_dim)
        initial_h = self._init_h(mean_encoder)

        return initial_h

    @overrides
    def _decode(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get data from state
        metadata = state['metadata']
        x = state['x']
        h = state['h']
        label = state['label']
        mask = state['mask']
        
        # Get actual size of current batch
        # Use main features to find current batch_size
        local_batch_size = x[0].shape[0]

        # Sort data to be able to only compute relevent parts of the batch at each timestep
        # Shape: (batch_size)
        lengths = mask.sum(dim=1)
        # Shape: (batch_size) (batch_size)
        sorted_lengths, indices = lengths.sort(dim=0, descending=True)
        # Computing last timestep isn't necessary with labels since last timestep is eos token or pad token 
        timesteps = sorted_lengths[0] - 1

        # Shape: (batch_size, ?)
        # x is a list; Shape: (batch_size, height * width, encoder_dim), (batch_size, height * width, encoder_dim)
        # Shape: (batch_size, decoder_dim)
        # Shape: (batch_size, timesteps)
        # Shape: (batch_size, timesteps)
        metadata = [metadata[i] for i in indices]
        # Sort indices of values in list separately
        x = [x[0][indices], x[1][indices]]
        h = h[indices]
        label = label[indices]        
        mask = mask[indices]
        
        # Shape: (batch_size, 1)
        predicted_indices = torch.LongTensor([[self._start_index]] * local_batch_size).to(device).view(-1, 1)
        
        # Shape: (batch_size, timesteps, vocab_size)
        predictions = torch.zeros(local_batch_size, timesteps, self._decoder.get_output_dim(), device=device)
        # Attention weights is a tuple
        attention_weights = (torch.zeros(local_batch_size, timesteps, self._encoder.get_feature_map_size(), device=device), torch.zeros(local_batch_size, timesteps, self._encoder.get_feature_map_size() * 2 * 2, device=device))
        sum_attention_weights_0 = torch.zeros(local_batch_size, self._encoder.get_feature_map_size(), device=device)
        sum_attention_weights_1 = torch.zeros(local_batch_size, self._encoder.get_feature_map_size() * 2 * 2, device=device)

        for t in range(timesteps):
            # Shape: (batch_offset)
            batch_offset = sum([l > t for l in sorted_lengths.tolist()])

            # Only compute data in valid timesteps
            # x_t is a list; Shape: (batch_offset, height * width, encoder_dim), (batch_offset, height * width, encoder_dim)
            # Shape: (batch_offset, decoder_dim)
            # Shape: (batch_offset, decoder_dim)
            # Shape: (batch_offset, 1)
            x_t = [x[0][:batch_offset], x[1][:batch_offset]]
            h_t = h[:batch_offset]
            predicted_indices_t = predicted_indices[:batch_offset]
            
            # Decode timestep
            # Shape: (batch_size, decoder_dim) (batch_size, vocab_size), (batch_size, encoder_dim, 1), (batch_size, height * width)
            h, preds, attention_weight, sum_attention_weights_0, sum_attention_weights_1 = self._decoder(x_t, h_t, predicted_indices_t, sum_attention_weights_0, sum_attention_weights_1)
            
            # Get new predicted indices to pass into model at next timestep
            # Use teacher forcing if chosen
            if self._teacher_forcing and np.random.random() < self._scheduled_sampling_ratio:
                # Send next timestep's label to next timestep
                # Shape: (batch_size, 1)
                predicted_indices = label[:batch_offset, t + 1].view(-1, 1)
            else:
                # Shape: (batch_size, 1)
                predicted_indices = torch.argmax(preds, dim=1).view(-1, 1)
            
            # Save preds
            predictions[:batch_offset, t, :] = preds
            
            # Attention weights is a tuple
            attention_weights[0][:batch_offset, t, :] = attention_weight[0].view(-1, self._encoder.get_feature_map_size())
            attention_weights[1][:batch_offset, t, :] = attention_weight[1].view(-1, self._encoder.get_feature_map_size() * 2 * 2)

        # Update state and add logits
        state['metadata'] = metadata
        state['x'] = x
        state['h'] = h
        state['label'] = label
        state['mask'] = mask
        state['attention_weights'] = attention_weights
        state['logits'] = predictions
            
        return state

    def _beam_search_step(self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Group_size is batch_size * beam_size except for first decoding timestep where it is batch_size
        # Shape: (group_size, decoder_dim) (group_size, decoder_dim) (group_size, vocab_size)
        
        # Combine main and multiscale features
        x = [state['x_0'], state['x_1']]
        h, predictions, attention_weights, sum_attention_weights_0, sum_attention_weights_1 = self._decoder(x, state['h'], last_predictions, state['sum_attention_weights_0'], state['sum_attention_weights_1'])
    
        # Attention weights is a tuple with main and multiscale features
        if self._attention_weights is not None:
            attention_weights = (attention_weights[0].view(-1, self._beam_size, 1, self._encoder.get_feature_map_size()), attention_weights[1].view(-1, self._beam_size, 1, self._encoder.get_feature_map_size() * 2 * 2))
            self._attention_weights = (torch.cat([self._attention_weights[0], attention_weights[0][:, 0, :, :]], dim=1), torch.cat([self._attention_weights[1], attention_weights[1][:, 0, :, :]], dim=1))
        else:
            attention_weights = (attention_weights[0].view(-1, 1, self._encoder.get_feature_map_size()), attention_weights[1].view(-1, 1, self._encoder.get_feature_map_size() * 2 * 2))
            self._attention_weights = attention_weights

        # Update state
        # Shape: (group_size, decoder_dim)
        state['h'] = h
        
        state['sum_attention_weights_0'] = sum_attention_weights_0
        state['sum_attention_weights_0'] = sum_attention_weights_0

        # Run log_softmax over logit predictions
        # Shape: (group_size, vocab_size)
        log_preds = F.log_softmax(predictions, dim=1)
        state

        return log_preds, state

    @overrides
    def _beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get data from state
        x = state['x']
        h = state['h']
        
        # x is a list; use main features; Get actual size of current batch
        local_batch_size = x[0].shape[0]

        # Beam search wants initial preds of shape: (batch_size)
        # Shape: (batch_size)
        initial_indices = torch.LongTensor([[self._start_index]] * local_batch_size).to(device).view(-1)        
        sum_attention_weights_0 = torch.zeros(local_batch_size, self._encoder.get_feature_map_size(), device=device)
        sum_attention_weights_1 = torch.zeros(local_batch_size, self._encoder.get_feature_map_size() * 2 * 2, device=device)

        # Beam search requires tensors, not lists
        state = {'x_0': x[0], 'x_1': x[1], 'h': h, 'sum_attention_weights_0': sum_attention_weights_0, 'sum_attention_weights_1': sum_attention_weights_1}
        
        # Timesteps returned aren't necessarily max_timesteps
        # Shape: (batch_size, beam_size, timesteps), (batch_size, beam_size)
        
        self._attention_weights = None
        
        predictions, log_probabilities = self.beam_search.search(initial_indices, state, self._beam_search_step)

        # Only keep best predictions from beam search
        # Shape: (batch_size, timesteps)
        predictions = predictions[:, 0, :].view(local_batch_size, -1)
        
        return predictions

    @overrides
    def forward(self, metadata: object, img: torch.Tensor, label: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Encode the image
        # Shape: (batch_size, height * width, encoder_dim)
        x = self._encoder(img)

        state = {'metadata': metadata, 'x': x}
        # Compute loss on train and val
        if label is not None:
            # Initialize h and c
            # Shape: (batch_size, decoder_dim)
            state['h'] = self._init_hidden(x)

            # Convert label dict to tensor since label isn't an input to the model and get mask
            # Shape: (batch_size, timesteps)
            state['mask'] = get_text_field_mask(label).to(device)
            # Shape: (batch_size, timesteps)
            state['label'] = label['tokens']

            # Decode encoded image and get loss on train and val
            state = self._decode(state)

            # Loss shouldn't be computed on start token
            state['mask'] = state['mask'][:, 1:].contiguous()
            state['target'] = state['label'][:, 1:].contiguous()

            # Compute cross entropy loss
            state['loss'] = sequence_cross_entropy_with_logits(state['logits'], state['target'], state['mask'])
            # Doubly stochastic regularization
            # Can't use doubly stochastic regularization with multiscale features
            # state['loss'] += ((1 - torch.sum(state['attention_weights'], dim=1)) ** 2).mean()

        # Decode encoded image with beam search on val and test
        if not self.training:
            # (Re)initialize h
            state['h'] = self._init_hidden(state['x'])
            
            # Run beam search
            state['out'] = self._beam_search(state)

            # Save attention weights
            # Predictor needs tensors, not tuple
            state['main_attention_weights'] = self._attention_weights[0]
            state['multiscale_attention_weights'] = self._attention_weights[1]

            # Compute validation scores
            if 'label' in state:
                self._bleu(state['out'], state['target'])
                self._exprate(state['out'], state['target'])
            
        # Set out to logits while training
        else:
            state['out'] = state['logits']
            
        return state
