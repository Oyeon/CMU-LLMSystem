# pipeline/model_parallel.py

import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel
from typing import Union, Optional, Tuple

from .pipe import Pipe
from .partition import WithDevice, _retrieve_device
from .model import GPT2ModelCustom, GPT2LMHeadModelCustom

class GPT2ModelParallel(GPT2ModelCustom):
    def __init__(self, config):
        super().__init__(config)

    # ASSIGNMENT 4.2
    def _prepare_pipeline_parallel(self, split_size=1):
        """
        Prepare the model for pipeline parallelism:
        1) Create an nn.Sequential from the GPT2Block layers (self.h).
        2) Wrap in Pipe(...), storing in self.h_pp.
        3) set self.pipeline_parallel = True
        """
        blocks = []
        for blk in self.h:
            # each block returns (hidden_states, present, attentions?)
            # so we wrap it with a small ExtractHidden layer
            wrapper = nn.Sequential(
                blk,
                ExtractHidden()
            )
            blocks.append(wrapper)

        seq = nn.Sequential(*blocks)
        pipe = Pipe(seq, split_size=split_size)

        self.h_pp = pipe
        self.pipeline_parallel = True

class ExtractHidden(nn.Module):
    """
    A simple module that just extracts the first element of the tuple
    that a GPT2Block returns (i.e., the hidden_states).
    """
    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple):
            return x[0]
        return x

class GPT2LMHeadModelParallel(GPT2LMHeadModelCustom):
    def __init__(self, config):
        super().__init__(config, GPT2ModelParallel(config))

    def _prepare_pipeline_parallel(self, split_size=1):
        self.parallelize()  # places each block on a different GPU
        self.transformer._prepare_pipeline_parallel(split_size=split_size)

    def _finalize_pipeline_parallel(self):
        self.deparallelize()
        self.transformer.pipeline_parallel = False
