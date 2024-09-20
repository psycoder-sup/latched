# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Generator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from coremltools.models import MLModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class CoreMLForwardModel(nn.Module):
    """
    Model for the CoreML forward
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)[0]


# https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/generate.py


def load(model_path: str, tokenizer_path: str) -> Tuple[MLModel, PreTrainedTokenizerBase]:
    model: MLModel = MLModel(model_path)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def get_next_token(model: MLModel, prompt_token: np.ndarray) -> Generator[int, None, None]:
    def sample(logits: np.ndarray) -> int:
        return int(np.argmax(logits[0][-1], axis=-1))

    def inference(model: MLModel, input_ids: np.ndarray, num_past_tokens: int) -> np.ndarray:
        causal_mask: np.ndarray = np.triu(
            np.full(
                (1, 1, input_ids.shape[-1], num_past_tokens + input_ids.shape[-1]),
                fill_value=np.inf if num_past_tokens == 0 else 0,
            ),
            k=1,
        ).astype(np.float16)
        outputs: Dict[str, np.ndarray] = model.predict(
            data={"inputIds": input_ids, "causalMask": causal_mask},
            state=kv_cache_state,
        )
        return outputs["logits"]

    kv_cache_state = model.make_state()
    logits: np.ndarray = inference(model, input_ids=prompt_token, num_past_tokens=0)
    token: int = sample(logits=logits)
    num_past_tokens: int = prompt_token.shape[-1]

    while True:
        yield token
        logits = inference(model, input_ids=np.array([[token]], dtype=np.int32), num_past_tokens=num_past_tokens)
        token = sample(logits=logits)
        num_past_tokens += 1


def generate(
    model: MLModel,
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int,
) -> str:
    prompt_tokens: np.ndarray = tokenizer(prompt, return_tensors="np").input_ids
    extended_tokens: List[int] = []
    for i, token in enumerate(get_next_token(model, prompt_token=prompt_tokens.astype(np.int32))):
        if token == tokenizer.eos_token_id or i == max_new_tokens:
            break
        extended_tokens.append(token)

    return tokenizer.decode(prompt_tokens[0].tolist() + extended_tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    model, tokenizer = load(args.model_path, args.tokenizer_path)

    generated_text: str = generate(model, args.prompt, tokenizer, args.max_new_tokens)
    print(generated_text)
