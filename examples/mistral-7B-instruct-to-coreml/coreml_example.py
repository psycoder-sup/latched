# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "mistralai/Mistral-7B-Instruct-v0.3"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


class MistralModelForCausalLM(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask).logits


test_input = tokenizer("Hello, how are you?", return_tensors="pt")

print(test_input)
