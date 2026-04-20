"""Backbone-agnostic VLM wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def _numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    return [Image.fromarray(img.astype(np.uint8)) for img in images]


class BaseVLM(nn.Module):
    """Holds a HF VLM + processor and common pre/post-processing helpers.

    Uses ``AutoModelForImageTextToText`` (confirmed to register Qwen3-VL in
    iter 3's hello-VLM smoke test).
    """

    def __init__(
        self,
        vlm_name: str,
        min_pixels: int,
        max_pixels: int,
        attn_implementation: str = "flash_attention_2",
        dtype: torch.dtype = torch.float16,
        device_map: str = "cuda",
    ) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(
            vlm_name,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        # Downstream code uses tokenizer.pad_token_id to build attention
        # masks and the generated-span mask for Inv-4. If the tokenizer
        # didn't declare one, fall back to eos_token — zero would be a
        # valid content token in some SentencePiece vocabs and would
        # silently mask real generated tokens out of the logprob sum.
        if self.processor.tokenizer.pad_token_id is None:
            if self.processor.tokenizer.eos_token_id is None:
                raise RuntimeError(
                    f"{vlm_name} tokenizer has neither pad_token_id nor eos_token_id; "
                    "cannot build attention_mask / span_mask safely."
                )
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model = AutoModelForImageTextToText.from_pretrained(
            vlm_name,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )

    def preprocess_obs_and_text(
        self,
        obs: torch.Tensor,
        text_prompts: list[str],
        add_generation_prompt: bool = True,
    ) -> Any:
        pil_images = _numpy_to_pil(obs.cpu().numpy())
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            for t in text_prompts
        ]
        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        return inputs

    @staticmethod
    def last_hidden_state(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_token_indices, :]

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
