import os
import random
from typing import Dict, Tuple, List
from overrides import overrides
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import allennlp

from allennlp.common import Registrable, Params

from allennlp.data import Vocabulary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE, self).__init__()
        
        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)
    
    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        
        return x

class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()
        
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)
        
    def forward(self, x):
        input_x = x
        
        x = self.conv(x)
        x = torch.sigmoid(x)
        
        x = torch.mul(input_x, x)
        
        return x

class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()
        
        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)
        
    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)
        
        x = torch.add(cSE, sSE)
        
        return x

class WAPConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super(WAPConv, self).__init__()
        
        self._dropout = dropout
        
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        if self._dropout:
            x = F.dropout(x, 0.2)
        
        return x

class WAPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super(WAPBlock, self).__init__()
        
        self.conv_1 = WAPConv(in_ch, out_ch, dropout=dropout)
        self.conv_2 = WAPConv(out_ch, out_ch, dropout=dropout)
        self.conv_3 = WAPConv(out_ch, out_ch, dropout=dropout)
        self.conv_4 = WAPConv(out_ch, out_ch, dropout=dropout)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        
        x = self.pool(x)

        return x

# Can't be pretrained; param is only for compatibility
def WAPBackbone(pretrained: bool = False):
    model = nn.Sequential(
        WAPBlock(3, 32),
        WAPBlock(32, 64),
        WAPBlock(64, 64),
        WAPBlock(64, 128, dropout=True)
    )
    
    return model

# From https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# Can't be pretrained; param is only for compatibility
def densenet(pretrained: bool = False):
    model =  torchvision.models.DenseNet(growth_rate=24, block_config=(32, 32, 32), num_init_features=48)
    
    return model

class Encoder(nn.Module, Registrable):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        
        self._pretrained = pretrained
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_output_dim(self) -> int:
        raise NotImplementedError()
        
    def get_feature_map_size(self) -> int:
        raise NotImplementedError()
        
@Encoder.register('backbone')
class BackboneEncoder(Encoder):
    def __init__(self, encoder_type: str = 'renset18', encoder_height: int = 4, encoder_width: int = 16, pretrained: bool = False, custom_in_conv: bool = False) -> None:
        super().__init__(pretrained=pretrained)
        
        self._encoder_type = encoder_type
        
        self._encoder_height = encoder_height
        self._encoder_width = encoder_width
        
        self._custom_in_conv = custom_in_conv
        
        self._backbones = {
            'vgg16': {
                'model': torchvision.models.vgg16,
                'encoder_dim': 512
            },
            'resnet18': { # 4 x 16
                'model': torchvision.models.resnet18,
                'encoder_dim': 512
            },
            'resnet50': { # 4 x 16
                'model': torchvision.models.resnet50,
                'encoder_dim': 2048
            },
            'densenet': { # 8 x 32
                'model': densenet,
                'encoder_dim': 1356
            },
            'WAP': { # 8 x 32
                'model': WAPBackbone,
                'encoder_dim': 128
            },
            'Im2latex': { # 14 x 62
                'model': Im2latexBackbone,
                'encoder_dim': 512
            },
            'smallResnet18': { # 8 x 32
                'model': torchvision.models.resnet18,
                'encoder_dim': 256
            },
        }
        
        self._backbone = self._backbones[self._encoder_type]['model'](pretrained=self._pretrained)
        self._encoder_dim = self._backbones[self._encoder_type]['encoder_dim']
        
        if self._custom_in_conv:
            self._backbone._modules['conv1'] = nn.Conv2d(3, 64, 3, padding=1)
        
        modules = list(self._backbone.children())
        
        if self._encoder_type == 'densenet':
            modules = modules[0][:-1]
        elif self._encoder_type == 'vgg16':
            modules = modules[:-1]
        elif self._encoder_type == 'smallResnet18':
            modules = modules[:-3]
        elif self._encoder_type == 'resnet18' or self._encoder_type == 'resnet50':
            modules = modules[:-2]

            # Add SCSE between resnet blocks
            modules = nn.Sequential(
                *modules[:5],
                SCSE(64, 16),
                *modules[5],
                SCSE(128, 16),
                *modules[6],
                SCSE(256, 16),
                *modules[7],
                SCSE(512, 16)
            )

        self._encoder = nn.Sequential(
            *modules,
            nn.AdaptiveAvgPool2d((self._encoder_height, self._encoder_width))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode image
        x = self._encoder(x)

        # Flatten image
        # Shape: (batch_size, height * width, encoder_dim)
        x = x.view(x.shape[0], -1, x.shape[1])

        return x

    def get_output_dim(self) -> int:
        return self._encoder_dim
    
    def get_feature_map_size(self) -> int:
        return self._encoder_height * self._encoder_width
        
@Encoder.register('lstm')
class LstmEncoder(Encoder):
    # Don't set hidden_size manually
    def __init__(self, encoder: Encoder, hidden_size: int = 512, layers: int = 1, bidirectional: bool = False) -> None:
        super().__init__(pretrained=False)

        self._encoder = encoder
        
        self._hidden_size = hidden_size
        self._layers = layers
        self._bidirectional = bidirectional
        
        self._lstm = nn.LSTM(input_size=self._encoder.get_output_dim(), hidden_size=self._hidden_size, num_layers=self._layers, batch_first=True, 
                             bidirectional=self._bidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode image
        # Shape: (batch_size, height * width, encoder_dim)
        x = self._encoder(x)
        
        # Encode encoded feature map with (bi)lstm
        # Shape: (batch_size, height * width, num_directions * hidden_size)
        x, _ = self._lstm(x)

        if self._bidirectional:
            # Shape: (batch_size, height * width, num_directions, hidden_size)
            x = x.view(-1, x.shape[1], 2, self._hidden_size)

            # Add directions and reverse bidirectional part
#             x = x[:, :, 0, :] + torch.from_numpy(np.flip(x[:, :, 1, :].detach().cpu().numpy(), axis=-1).copy()).to(device)
            x = x[:, :, 0, :] + x[:, :, 1, :]

        return x
    
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BasicTextFieldEmbedder':  # type: ignore
        encoder = params.pop("encoder")
        layers = params.pop("layers")
        bidirectional = params.pop("bidirectional")
        
        encoder = Encoder.from_params(vocab=vocab, params=encoder)
        
        return cls(encoder, encoder._encoder_dim, layers, bidirectional)

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_size
    
    def get_feature_map_size(self) -> int:
        return self._encoder._encoder_height * self._encoder._encoder_width
    
class Im2latexBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, padding: int, bn: bool = True, pool: bool = False, pool_stride: Tuple[int, int] = None):
        super(Im2latexBlock, self).__init__()
        
        self._bn = bn
        self._pool = pool

        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1 if padding else 0)
        
        if self._bn:
            self.bn = nn.BatchNorm2d(out_ch)
            
        if self._pool:
            self.pool = nn.MaxPool2d(pool_stride, pool_stride)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        if self._bn:
            x = self.bn(x)

        x = F.relu(x)

        if self._pool:
            x = self.pool(x)

        return x

# Can't be pretrained; param is only for compatibility
def Im2latexBackbone(pretrained: bool = False):
    model = nn.Sequential(
        Im2latexBlock(3, 64, 1, False, True, (2, 2)),
        Im2latexBlock(64, 128, 1, False, True, (2, 2)),
        Im2latexBlock(128, 256, 1, True, False),
        Im2latexBlock(256, 256, 1, False, True, (1, 2)),
        Im2latexBlock(256, 512, 1, True, True, (2, 1)),
        Im2latexBlock(512, 512, 0, True, False)
    )
    
    return model

@Encoder.register('Im2latex')
class Im2latexEncoder(Encoder):
    # Don't set hidden_size manually
    def __init__(self, encoder: Encoder, hidden_size: int = 512, layers: int = 1, bidirectional: bool = False) -> None:
        super().__init__(pretrained=False)
       
        self._hidden_size = hidden_size
        self._layers = layers
        self._bidirectional = bidirectional
        
        self._num_directions = 2 if self._bidirectional else 1
        
        self._encoder = encoder
        
        self._row_encoder = nn.GRU(input_size=self._encoder.get_output_dim(), hidden_size=self._hidden_size, num_layers=self._layers, batch_first=True, 
                                    bidirectional=self._bidirectional)
        
        self._positional_embeddings = nn.Embedding(self._encoder._encoder_height, self._hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode image
        # Shape: (batch_size, height * width, encoder_dim)
        x = self._encoder(x)
        
        # Shape: (batch_size, encoder_dim, height, width)
        x = x.view(-1, self._encoder._encoder_dim, self._encoder._encoder_height, self._encoder._encoder_width)

        # Shape: (batch_size, hidden_size, height, width)
        encoded_rows = torch.zeros((x.shape[0], self._hidden_size, self._encoder._encoder_height, self._encoder._encoder_width), device=device)
        
        # Go over each row
        for i in range(x.shape[2]):
            # Get row
            # Shape: (batch_size, width, encoder_dim)
            row = x[:, :, i].transpose(1, 2)
            
            # Get positional embeddings for row
            # Shape: (1, hidden_size)
            positional_embedding = self._positional_embeddings(torch.LongTensor([i]).to(device))

            # Duplicate positional embeddings for each element in batch
            # Shape: (layers * num_directions, batch_size, hidden_size)
            positional_embedding = positional_embedding.view(1, 1, self._hidden_size).repeat(self._layers * self._num_directions, x.shape[0], 1)
            
            # Encode row
            # Shape: (batch_size, width, num_directions * hidden_size)
            encoded_row, _ = self._row_encoder(row, positional_embedding)
            
            if self._bidirectional:
                # Shape: (batch_size, width, 2, hidden_size)
                encoded_row = encoded_row.view(-1, encoded_row.shape[1], 2, self._hidden_size)

                # Add bidirectional directions
                # Shape: (batch_size, width, hidden_size)
                encoded_row = encoded_row[:, :, 0, :] + encoded_row[:, :, 1, :]
                # Reverse bidirectional direction
#                 encoded_row = encoded_row[:, :, 0, :] + torch.from_numpy(np.flip(encoded_row[:, :, 1, :].detach().cpu().numpy(), axis=-1).copy()).to(device)

            # Shape: (batch_size, hidden_size, width)
            encoded_rows[:, :, i, :] = encoded_row.transpose(1, 2)

        # Shape: (batch_size, height * with, hidden_size)
        x = encoded_rows.view(-1, self._encoder.get_feature_map_size(), self._hidden_size)
    
        return x

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BasicTextFieldEmbedder':  # type: ignore
        encoder = params.pop("encoder")
        layers = params.pop("layers")
        bidirectional = params.pop("bidirectional")
        
        encoder = Encoder.from_params(vocab=vocab, params=encoder)
        
        return cls(encoder, encoder._encoder_dim, layers, bidirectional)

    def get_output_dim(self) -> int:
        return self._hidden_size
    
    def get_feature_map_size(self) -> int:
        return self._encoder._encoder_height * self._encoder._encoder_width

# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(int(in_planes), int(out_planes), kernel_size=3, stride=stride,padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(int(in_planes), int(out_planes), kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
# From https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, pool=True):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        if pool == True:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

# pretrained is only for compatibility
class MultiscaleDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.2, num_classes=1000, pretrained=False):

        super(MultiscaleDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            pool = True
            if i == len(block_config) - 1:
                pool = False
            
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, pool=pool)
                
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2

            # Add SCSE
            scse = nn.Sequential(SCSE(num_features, 16))
            self.features.add_module('scse%d' % (i + 1), scse)
            
        # Multiscale branch

        self.main_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            _DenseBlock(num_layers=32, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate),
            SCSE(1356, 16)
        )

        self.multiscale_branch = nn.Sequential(
            _DenseBlock(num_layers=16, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate),
            SCSE(972, 16)
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        main_features = self.main_branch(features)
        multiscale_features = self.multiscale_branch(features)

        return [main_features, multiscale_features]

@Encoder.register('multiscale')
class MultiscaleEncoder(Encoder):
    def __init__(self, encoder_type: str = 'renset18', encoder_height: int = 4, encoder_width: int = 16, pretrained: bool = False) -> None:
        super().__init__(pretrained=pretrained)
        
        self._encoder_type = encoder_type
        
        self._encoder_height = encoder_height
        self._encoder_width = encoder_width
        
        self._backbones = {
            'resnet18': {
                'model': torchvision.models.resnet18,
                'encoder_dim': [512]
            },
            'resnet50': {
                'model': torchvision.models.resnet50,
                'encoder_dim': [2048]
            },
            'densenet': {
                'model': MultiscaleDenseNet,
                'encoder_dim': [1356, 972]
            }
        }
                
        if self._encoder_type == 'resnet18' or self._encoder_type == 'resnet50':
            self._backbone = self._backbones[self._encoder_type]['model'](pretrained=self._pretrained)
            self._encoder_dim = self._backbones[self._encoder_type]['encoder_dim'][0]

            # Common conv blocks
            self._encoder = nn.Sequential(
                *list(self._backbone.children())[:-3]
            )

            # Last conv block
            self._main_branch = nn.Sequential(*list(self._backbone.children())[-3])

            # Uses 1x1 convs to convert identity to correct num of channels
            self._identity_conv = nn.Sequential(
                conv1x1(self._encoder_dim / 2, self._encoder_dim),
                nn.BatchNorm2d(self._encoder_dim),
            )

            # Last conv block without pool and not pretrained
            self._multiscale_branch = nn.Sequential(
                BasicBlock(self._encoder_dim / 2, self._encoder_dim, downsample=self._identity_conv),
                BasicBlock(self._encoder_dim, self._encoder_dim)
            )
        else:
            self._backbone = self._backbones[self._encoder_type]['model'](growth_rate=24, block_config=(32, 32), num_init_features=48)
            self._encoder_dim = self._backbones[self._encoder_type]['encoder_dim'][0]

            self._encoder = self._backbone

    def forward(self, x: torch.Tensor):
        # Encode image through common conv blocks
        # Shape: (batch_size, channels, height * 2, width * 2)
        x = self._encoder(x)

        if self._encoder_type == 'resnet18' or self._encoder_type == 'resnet50':
            # Shape: (batch_size, channels, height, width)
            main_features = self._main_branch(x)

            # Get multiscale features
            # Shape: (batch_size, channels, height * 2, width * 2)
            multiscale_features = self._multiscale_branch(x)            
        else:
            main_features, multiscale_features = x[0], x[1]
            
        # Flatten features
        # Shape: (batch_size, height * width, encoder_dim)
        main_features = main_features.view(main_features.shape[0], -1, main_features.shape[1])
        # Shape: (batch_size, height * 2 * width * 2, encoder_dim)
        multiscale_features = multiscale_features.view(multiscale_features.shape[0], -1, multiscale_features.shape[1])

        return [main_features, multiscale_features]
    
    def get_output_dim(self) -> int:
        return self._encoder_dim
    
    def get_feature_map_size(self) -> int:
        return self._encoder_height * self._encoder_width

#DEPRECATED
@Encoder.register('multiscale-lstm')
class LstmEncoder(Encoder):
    # Don't set hidden_size manually
    def __init__(self, encoder: Encoder, hidden_size: int = 256, layers: int = 1, bidirectional: bool = False) -> None:
        super().__init__(pretrained=False)

        self._encoder = encoder
        
        self._hidden_size = hidden_size
        self._layers = layers
        self._bidirectional = bidirectional
        
        self._lstm = nn.LSTM(input_size=1356, hidden_size=1356, num_layers=self._layers, batch_first=True, 
                             bidirectional=self._bidirectional)
        
        self._lstm2 = nn.LSTM(input_size=972, hidden_size=972, num_layers=self._layers, batch_first=True, 
                             bidirectional=self._bidirectional)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode image
        # Shape: (batch_size, height * width, encoder_dim)
        x = self._encoder(x)
        
        # Encode encoded main and dense feature map with (bi)lstm
        # Shape: (batch_size, height * width, num_directions * hidden_size)
        x_1, _ = self._lstm(x[0])
        x_2, _ = self._lstm2(x[1])

        if self._bidirectional:
            # Shape: (batch_size, height * width, num_directions, hidden_size)
            x_1 = x_1.view(-1, x_1.shape[1], 2, self._hidden_size)
            x_2 = x_2.view(-1, x_2.shape[1], 2, self._hidden_size)

            # Add directions and reverse bidirectional part
#             x_1 = x_1[:, :, 0, :] + torch.from_numpy(np.flip(x_1[:, :, 1, :].detach().cpu().numpy(), axis=-1).copy()).to(device)
#             x_2 = x_2[:, :, 0, :] + torch.from_numpy(np.flip(x_2[:, :, 1, :].detach().cpu().numpy(), axis=-1).copy()).to(device)

            x_1 = x_1[:, :, 0, :] + x_1[:, :, 1, :]
            x_2 = x_2[:, :, 0, :] + x_2[:, :, 1, :]

        return [x_1, x_2]
    
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BasicTextFieldEmbedder':  # type: ignore
        encoder = params.pop("encoder")
        layers = params.pop("layers")
        bidirectional = params.pop("bidirectional")
        
        encoder = Encoder.from_params(vocab=vocab, params=encoder)
        
        return cls(encoder, encoder._encoder_dim, layers, bidirectional)
    
    def get_output_dim(self) -> int:
        return self._encoder._encoder_dim
    
    def get_feature_map_size(self) -> int:
        return self._encoder._encoder_height * self._encoder._encoder_width
