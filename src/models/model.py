import torch
import torch.nn as nn
import numpy as np

from IPython import embed
from torch.distributions.categorical import Categorical
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from src.utils.utils import numpy_to_pil

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def default_target_modules():
    target_modules = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "attn.qkv",
        "attn.proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    # Remove potential duplicates while preserving order
    seen = set()
    target_modules_unique = [m for m in target_modules if not (m in seen or seen.add(m))]
    return target_modules_unique

class BaseVLM(nn.Module):
    """
    Base class for VLM-based actor/critic. Provides shared logic for processor/model init,
    prompt/image handling, and target_modules setup.
    """
    def __init__(self, vlm_name: str, max_new_tokens: int = 128):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            vlm_name,
            trust_remote_code=True,
            min_pixels=210 * 160 * 3,
            max_pixels=210 * 160 * 3,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name,
            torch_dtype=torch.bfloat16,
            # dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

    def preprocess_obs_and_text(self, obs, text_prompts):
        # obs: [B, ...], text_prompts: list[str] (len B)
        pil_images = numpy_to_pil(obs.cpu().numpy())
        texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
            tokenize=False, add_generation_prompt=True
        ) for t in text_prompts]
        inputs = self.processor(
            text=texts, images=pil_images, return_tensors="pt", padding=True,
        ).to(self.model.device)
        return inputs

    def set_target_modules(self):
        self.target_modules = default_target_modules()
        return self.target_modules

    def last_hidden_state(self, hidden_states, attention_mask):
        """Calculates the hidden state of the last non-padding token."""
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden_state = hidden_states[batch_indices, last_token_indices, :]
        return last_hidden_state

# --- CRITIC HEAD ---
class CriticHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def forward(self, hidden):
        return self.net(hidden)

# --- SHARED ACTOR-CRITIC ---
class SharedActorCriticVLM(BaseVLM):
    def __init__(self, vlm_name: str, max_new_tokens: int = 128):
        super().__init__(vlm_name, max_new_tokens)
        hidden_size = self.model.config.hidden_size
        self.critic = CriticHead(hidden_size).to(self.model.dtype)

    def get_value(self, obs, prompt_text):
        inputs = self.preprocess_obs_and_text(obs, prompt_text)
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )
        last_hidden = self.last_hidden_state(outputs.hidden_states[-1], inputs['attention_mask'])
        return self.critic(last_hidden)

    def get_action_and_value(self, obs=None, text_prompts=None, action_ids=None, prompt_lens=None):
        batch_size = len(text_prompts)
        inputs = self.preprocess_obs_and_text(obs, text_prompts)
        pixel_values = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw

        # --- Generation ---
        if action_ids is None:
            full_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            generated_texts = self.processor.batch_decode(
                full_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            prompt_lens = torch.tensor([inputs.input_ids.shape[1]] * batch_size, device=self.model.device)
        else:
            full_ids = action_ids

        full_attention_mask = (full_ids != self.processor.tokenizer.pad_token_id).long()
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            image_grid_thw=image_grid_thw,
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        logits = outputs.logits
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs_all, 2, full_ids.unsqueeze(-1)).squeeze(-1)

        indices = torch.arange(full_ids.shape[1], device=full_ids.device)
        action_token_mask = indices[None, :] >= (prompt_lens)[:, None]
        pad_mask = (full_ids != self.processor.tokenizer.pad_token_id)
        final_mask = action_token_mask & pad_mask

        masked_log_probs = log_probs * final_mask

        # critic uses last input token, not last token of full_ids!
        last_input_token_indices = prompt_lens.to(full_attention_mask.device) - 1  # shape [B]
        batch_indices = torch.arange(full_ids.size(0), device=full_ids.device)
        last_hidden_states = outputs.hidden_states[-1]  # [B, T, hidden]
        critic_input_reps = last_hidden_states[batch_indices, last_input_token_indices, :]  # [B, hidden]
        value = self.critic(critic_input_reps)

        entropy = Categorical(logits=logits).entropy()
        masked_entropy = entropy * final_mask

        if action_ids is None:
            return masked_log_probs, value, full_ids, full_attention_mask, prompt_lens, generated_texts
        else:
            return masked_log_probs, value, masked_entropy, final_mask

# --- SEPARATE ACTOR ---
class ActorVLM(BaseVLM):
    def __init__(self, vlm_name: str, max_new_tokens: int = 128):
        super().__init__(vlm_name, max_new_tokens)

    def get_action(self, obs=None, text_prompts=None, action_ids=None, prompt_lens=None):
        batch_size = len(text_prompts)
        inputs = self.preprocess_obs_and_text(obs, text_prompts)
        pixel_values = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw

        # --- Generation ---
        if action_ids is None:
            full_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            generated_texts = self.processor.batch_decode(
                full_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            prompt_lens = torch.tensor([inputs.input_ids.shape[1]] * batch_size, device=self.model.device)
        else:
            full_ids = action_ids

        full_attention_mask = (full_ids != self.processor.tokenizer.pad_token_id).long()
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            image_grid_thw=image_grid_thw,
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        logits = outputs.logits
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs_all, 2, full_ids.unsqueeze(-1)).squeeze(-1)

        indices = torch.arange(full_ids.shape[1], device=full_ids.device)
        action_token_mask = indices[None, :] >= (prompt_lens)[:, None]
        pad_mask = (full_ids != self.processor.tokenizer.pad_token_id)
        final_mask = action_token_mask & pad_mask

        masked_log_probs = log_probs * final_mask

        entropy = Categorical(logits=logits).entropy()
        masked_entropy = entropy * final_mask

        if action_ids is None:
            return masked_log_probs, full_ids, full_attention_mask, prompt_lens, generated_texts
        else:
            return masked_log_probs, masked_entropy, final_mask

# --- SEPARATE CRITIC ---
class CriticVLM(BaseVLM):
    def __init__(self, vlm_name: str):
        super().__init__(vlm_name)
        hidden_size = self.model.config.hidden_size
        self.critic = CriticHead(hidden_size).to(self.model.dtype)

    def get_value(self, obs, prompt_text):
        inputs = self.preprocess_obs_and_text(obs, prompt_text)
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )
        last_hidden = self.last_hidden_state(outputs.hidden_states[-1], inputs['attention_mask'])
        return self.critic(last_hidden)