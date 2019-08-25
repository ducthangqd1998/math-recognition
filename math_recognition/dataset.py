import os
import random
from typing import Dict, Tuple, List
from overrides import overrides

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

import spacy

import allennlp

from allennlp.common.util import START_SYMBOL, END_SYMBOL, get_spacy_model

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ArrayField, TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, CharacterTokenizer, WordTokenizer

@Tokenizer.register("latex")
class LatexTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def _tokenize(self, text):        
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        text = text.replace('{', ' { ')
        text = text.replace('}', ' } ')
#         text = text.replace('$', ' $ ')
        text = text.replace('$', '')
        text = text.replace('_', ' _ ')
        text = text.replace('^', ' ^ ')
        text = text.replace('+', ' + ')
        text = text.replace('-', ' - ')
        text = text.replace('/', ' / ')
        text = text.replace('*', ' * ')
        text = text.replace('=', ' = ')
        text = text.replace('[', ' [ ')
        text = text.replace(']', ' ] ')
        text = text.replace('|', ' | ')
        text = text.replace('!', ' ! ')
        text = text.replace(',', ' , ')
        
        text = text.replace('\\', ' \\')
        
        text = text.replace('0', ' 0 ')
        text = text.replace('1', ' 1 ')
        text = text.replace('2', ' 2 ')
        text = text.replace('3', ' 3 ')
        text = text.replace('4', ' 4 ')
        text = text.replace('5', ' 5 ')
        text = text.replace('6', ' 6 ')
        text = text.replace('7', ' 7 ')
        text = text.replace('8', ' 8 ')
        text = text.replace('9', ' 9 ')
        
        text2 = ''
        for word in text.split():
            if len(word) > 1:
                if word[0] != '\\':
                    for char in word:
                        text2 += f' {char}'
                else:
                    text2 += f' {word}'
            else:
                text2 += f' {word}'

        return [Token(token) for token in text2.split()]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = self._tokenize(text)

        return tokens
    
# From https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
def resize(im, desired_size):

    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im

@DatasetReader.register('CROHME')
class CROHMEDatasetReader(DatasetReader):
    def __init__(self, root_path: str, tokenizer: Tokenizer, height: int = 512, width: int = 512, lazy: bool = True,
                 subset: bool = False) -> None:
        super().__init__(lazy)
        
        self.mean = 0.4023
        self.std = 0.4864
        
        self.root_path = root_path
        self.height = height
        self.width = width
        self.subset = subset
        
        self._tokenizer = tokenizer
        self._token_indexer = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file: str):
        df = pd.read_csv(os.path.join(self.root_path, file))
        if self.subset:
            df = df.loc[:16]

        for _, row in df.iterrows():
            img_id = row['id']
            
            if 'label' in df.columns:
                label = row['label']
                yield self.text_to_instance(file, img_id, label)
            else:
                yield self.text_to_instance(file, img_id)
            
    @overrides
    def text_to_instance(self, file: str, img_id: int, label: str = None) -> Instance:
        sub_path = file.split('/')[0]
        path = os.path.join(self.root_path, sub_path, 'data', f'{img_id}.png')

        img = (1 - plt.imread(path)[:,:,0])
        img = img.reshape(1, img.shape[0], img.shape[1])
        img = np.concatenate((img, img, img))
        img = cv2.resize(img.transpose(1, 2, 0), (self.width, self.height)).transpose(2, 0, 1)
        img = np.rint(img)
    
        fields = {}
        fields['metadata'] = MetadataField({'path': path})
        fields['img'] = ArrayField(img)
        
        if label is not None:
            label = self._tokenizer.tokenize(label)

            label.insert(0, Token(START_SYMBOL))
            label.append(Token(END_SYMBOL))
            
            fields['label'] = TextField(label, self._token_indexer)
        
        return Instance(fields)
