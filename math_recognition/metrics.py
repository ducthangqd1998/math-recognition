import os
import random
import subprocess
from typing import Dict, Tuple
from overrides import overrides

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import allennlp

from allennlp.common import Registrable, Params
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from allennlp.training.metrics import Metric, F1Measure, BLEU, BooleanAccuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# From https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/boolean_accuracy.py
@Metric.register("exprate")
class Exprate(Metric):
    def __init__(self, end_index: int, vocab) -> None:
        self._correct = 0.0
        self._total = 0.0
        self.vocab = vocab

        self._end_index = end_index

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor):
        predictions, targets = self.unwrap_to_tensors(predictions, targets)
        batch_size = predictions.size(0)

        # Shape: (batch_size, -1)
        predictions = predictions.view(batch_size, -1)
        # Shape: (batch_size, -1)
        targets = targets.view(batch_size, -1)

        # Get index of eos token in targets
        end_indices = (targets == self._end_index).nonzero()[:, 1]

        # Check if each prediction in batch is identical to target
        for i in range(batch_size):
            end_index = end_indices[i]

            # Shape: (1, -1)
            target = targets[i, :end_index]

            # Shape: (1, -1)
            prediction = predictions[i, :end_index]

            if torch.equal(prediction, target):
                self._correct += 1
            self._total += 1

    def get_metric(self, reset: bool = False):
        accuracy = float(self._correct) / float(self._total)
        if reset:
            self.reset()
        return {"exprate": accuracy}

    @overrides
    def reset(self):
        self._correct = 0.0
        self._total = 0.0
