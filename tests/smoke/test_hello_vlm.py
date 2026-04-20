"""Hello-VLM smoke: load Qwen/Qwen3.5-VL-0.8B, feed a synthetic image + prompt, generate a response.

Verifies:
  * Dependency graph resolves and imports.
  * Backbone loads (requires transformers from git per master-spec §3).
  * AutoProcessor handles multimodal inputs.
  * Generation path does not error.

This is NOT a correctness test for learning — it only proves the scaffold
infra is runnable. Master-spec §0 governs the response to any failure:
investigate → understand → iterate. Hard-fail only on genuine correctness
bugs (wrong shapes, NaN, import errors).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


MODEL_ID = os.environ.get("CLEANRL_VLM_SMOKE_MODEL", "Qwen/Qwen3.5-VL-0.8B")


@pytest.fixture(scope="module")
def synthetic_image() -> Image.Image:
    """A 256x256 RGB image with four distinct colored quadrants."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # top-left red, top-right green, bottom-left blue, bottom-right yellow
    arr[:128, :128] = [220, 20, 20]
    arr[:128, 128:] = [20, 200, 20]
    arr[128:, :128] = [20, 20, 220]
    arr[128:, 128:] = [230, 230, 20]
    return Image.fromarray(arr, mode="RGB")


@pytest.mark.tier1
@pytest.mark.gpu
@pytest.mark.timeout(600)
def test_hello_vlm_loads_and_generates(synthetic_image: Image.Image, tmp_path: Path) -> None:
    """Load Qwen3.5-VL-0.8B, feed a synthetic quadrant image + prompt, generate."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoProcessor

    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "List the dominant colors in each quadrant of this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[synthetic_image],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    decoded = processor.batch_decode(
        generated[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    # Artifact: dump the prompt + response under tmp_path for inspection on CI failure.
    (tmp_path / "hello_vlm_output.txt").write_text(
        f"PROMPT:\n{text}\n\nRESPONSE:\n{decoded}\n", encoding="utf-8"
    )

    # The ONLY assertion is that generation produced at least one token of output.
    # We do not assert the content (that's the ground-truth vision probe's job — Inv-15).
    assert decoded.strip(), "model produced empty output"
    assert len(generated[0]) > inputs.input_ids.shape[1], "no new tokens generated"
