
import os
import random
from typing import Dict, Tuple
from overrides import overrides
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import skimage
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import allennlp

from allennlp.common import Registrable, Params
from allennlp.common.util import START_SYMBOL, END_SYMBOL, JsonDict

from allennlp.data import DatasetReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.predictors.predictor import Predictor

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper # MIGHT USE FOR ABSTRACTION

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.beam_search import BeamSearch

from allennlp.training.metrics import F1Measure, BLEU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@Predictor.register('CROHME')
class MathPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        
        self._start_idx = np.random.randint(0, 100)
        self._counter = self._start_idx

    def dump_line(self, outputs: JsonDict) -> str:
        beam_search_preds = [self._model.vocab.get_token_from_index(i) for i in outputs['out']]
        preds = ' '.join(beam_search_preds)
        idx = preds.index('@end@')
        preds = preds[:idx]
        out = '\n\nBeam search pred: ' + preds + '\n'
        # out = preds + '\n'

        if 'label' in outputs:
            label = ' '.join([self._model.vocab.get_token_from_index(i) for i in outputs['label']])
            end_idx = label.index('@end@')
            label = label[8:end_idx]
            out += 'Gold: ' + label + '\n'
            # out += label + '\n'

        if 'logits' in outputs:
            logits = np.array(outputs['logits'])
            out += 'Logits: ' + str([self._model.vocab.get_token_from_index(np.argmax(logits[i])) for i in range(logits.shape[0])])

        # Save visualizations for first 10 preds
        if self._counter - self._start_idx < 10:
            img = plt.imread(outputs['metadata']['path'])
            img = cv2.resize(img, (512, 128))
            
            attention_weights = np.array(outputs['attention_weights'])
            timesteps = attention_weights.shape[0]
            
            fig=plt.figure(figsize=(20, 20))
            fig.tight_layout() 
            columns = 8
            rows = 10
            for i in range(1, timesteps + 1):
                ax = fig.add_subplot(rows, columns, i)
                ax.set_title(f'{beam_search_preds[i-1]}')

                plt.imshow(img)

                attention_weight = attention_weights[i-1].reshape(4, 16)
#                 attention_weight = attention_weights[i-1].reshape(8, 32)
                attention_weight = skimage.transform.pyramid_expand(attention_weight, upscale=32, sigma=8)
#                 attention_weight = skimage.transform.pyramid_expand(attention_weight, upscale=16, sigma=8)
                plt.imshow(attention_weight, alpha=0.8)
            
            save_path = 'visualization_' + outputs['metadata']['path'].split('/')[2] + f'_{self._counter}.png'
            fig.savefig(save_path)
            
            self._counter += 1
            
        return out

@Predictor.register('WAP')
class WAPPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        
        self._start_idx = np.random.randint(0, 100)
        self._counter = self._start_idx

    def dump_line(self, outputs: JsonDict) -> str:
        beam_search_preds = [self._model.vocab.get_token_from_index(i) for i in outputs['out']]
        out = '\n\nPred: ' + ' '.join(beam_search_preds) + '\n'

        if 'logits' in outputs:
            logits = np.array(outputs['logits'])
            out += 'Logits: ' + str([self._model.vocab.get_token_from_index(np.argmax(logits[i])) for i in range(logits.shape[0])]) + '\n'
                
        if 'label' in outputs:
            out += 'Gold: ' + ' '.join([self._model.vocab.get_token_from_index(i) for i in outputs['label']])
                
        # Save visualizations for first 10 preds
        if self._counter - self._start_idx < 10:
            img = plt.imread(outputs['metadata']['path'])
            img = cv2.resize(img, (512, 128))
            
            attention_weights = np.array(outputs['attention_weights'])
            timesteps = attention_weights.shape[0]

            fig=plt.figure(figsize=(20, 20))
            fig.tight_layout() 
            columns = 8
            rows = 10
            for i in range(1, timesteps + 1):
                ax = fig.add_subplot(rows, columns, i)
                ax.set_title(f'{beam_search_preds[i-1]}')
                
                plt.imshow(img)
                
                attention_weight = attention_weights[i-1].reshape(8, 32)
                attention_weight = skimage.transform.pyramid_expand(attention_weight, upscale=16, sigma=8)
                plt.imshow(attention_weight, alpha=0.8)
                
            save_path = 'visualization_' + outputs['metadata']['path'].split('/')[2] + f'_{self._counter}.png'
            fig.savefig(save_path)
            
            self._counter += 1
            
        return out

@Predictor.register('multiscale')
class MathPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        
        self._start_idx = np.random.randint(0, 100)
        self._counter = self._start_idx

    def dump_line(self, outputs: JsonDict) -> str:
        beam_search_preds = [self._model.vocab.get_token_from_index(i) for i in outputs['out']]
        out = '\n\nPred: ' + ' '.join(beam_search_preds) + '\n'

        if 'logits' in outputs:
            logits = np.array(outputs['logits'])
            out += 'Logits: ' + str([self._model.vocab.get_token_from_index(np.argmax(logits[i])) for i in range(logits.shape[0])]) + '\n'
                
        if 'label' in outputs:
            out += 'Gold: ' + ' '.join([self._model.vocab.get_token_from_index(i) for i in outputs['label']])
    
        # Save visualizations for first 10 preds
        if self._counter - self._start_idx < 10:
            img = plt.imread(outputs['metadata']['path'])
            img = cv2.resize(img, (512, 128))
            
            attention_weights = np.array(outputs['main_attention_weights'])
            timesteps = attention_weights.shape[0]

            fig=plt.figure(figsize=(20, 20))
            fig.tight_layout() 
            columns = 8
            rows = 10
            for i in range(1, timesteps + 1):
                ax = fig.add_subplot(rows, columns, i)
                ax.set_title(f'{beam_search_preds[i-1]}')
                
                plt.imshow(img)
                
                attention_weight = attention_weights[i-1].reshape(8, 32)
                attention_weight = skimage.transform.pyramid_expand(attention_weight, upscale=16, sigma=8)
                plt.imshow(attention_weight, alpha=0.8)
                
            save_path = 'visualization_' + outputs['metadata']['path'].split('/')[2] + f'_{self._counter}_main_branch.png'
            fig.savefig(save_path)

            attention_weights = np.array(outputs['multiscale_attention_weights'])
            timesteps = attention_weights.shape[0]

            fig=plt.figure(figsize=(20, 20))
            fig.tight_layout() 
            columns = 8
            rows = 10
            for i in range(1, timesteps + 1):
                ax = fig.add_subplot(rows, columns, i)
                ax.set_title(f'{beam_search_preds[i-1]}')
                
                plt.imshow(img)
                
                attention_weight = attention_weights[i-1].reshape(16, 64)
                attention_weight = skimage.transform.pyramid_expand(attention_weight, upscale=8, sigma=8)
                plt.imshow(attention_weight, alpha=0.8)
                
            save_path = 'visualization_' + outputs['metadata']['path'].split('/')[2] + f'_{self._counter}_multiscale_branch.png'
            fig.savefig(save_path)
            
            self._counter += 1
            
        return out