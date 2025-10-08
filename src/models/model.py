import torch
import torch.nn as nn
import numpy as np

from IPython import embed
from torch.distributions.categorical import Categorical
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, vlm_name: str, max_new_tokens: int = 128):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        
        self.processor = AutoProcessor.from_pretrained(vlm_name, trust_remote_code=True, min_pixels = 210 * 160 * 3, max_pixels = 210 * 160 * 3)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        
        hidden_size = self.model.config.hidden_size
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.critic.to(self.model.dtype)

    def get_value(self, input_ids, attention_mask):
        """Calculates value from the last non-padding token's hidden state."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = self.get_last_hidden_state(outputs.hidden_states[-1], attention_mask)
        return self.critic(last_hidden_state)

    def get_action_and_value(self, images, text_prompts, action_ids=None, prompt_lens=None):
        if action_ids is None:  # --- Generation Phase ---
            texts = [self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                tokenize=False, add_generation_prompt=True,
            ) for p in text_prompts]
            
            inputs = self.processor(
                text=texts, images=images, return_tensors="pt", padding=True,
            ).to(self.model.device)
            
            with torch.no_grad():
                full_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                    temperature=0.7, top_p=0.9,
                ) # shape (num_envs, seq_len) {seq_len = prompt_len + max_new_tokens}
                
            generated_texts = self.processor.batch_decode(
                full_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
             
            prompt_lens = torch.tensor([inputs.input_ids.shape[-1]] * len(images), device=self.model.device) # shape (num_envs)
        else:  # --- Update Phase ---
            full_ids = action_ids # shape (num_envs, seq_len) {seq_len = prompt_len + max_new_tokens}

        full_attention_mask = (full_ids != self.processor.tokenizer.pad_token_id).long() # shape (num_envs, seq_len)
        outputs = self.model(
            input_ids=full_ids, attention_mask=full_attention_mask, output_hidden_states=True
        )
        logits = outputs.logits # shape (num_envs, seq_len, vocab_size)
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)  # (num_envs, seq_len, vocab_size)
        log_probs = torch.gather(log_probs_all, 2, full_ids.unsqueeze(-1)).squeeze(-1)  # (num_envs, seq_len)

        # Mask out the prompt tokens, only keep generated tokens
        indices = torch.arange(full_ids.shape[1], device=full_ids.device) # shape (seq_len)
        action_token_mask = indices[None, :] >= (prompt_lens - 1)[:, None]
        pad_mask = (full_ids != self.processor.tokenizer.pad_token_id)
        final_mask = action_token_mask & pad_mask # shape (num_envs, seq_len)

        log_probs = log_probs * final_mask
        summed_log_probs = log_probs.sum(dim=-1)

        # get value. the critic currently takes the full prompt + generated tokens. {outputs.hidden_states[-1]  is of shape (num_envs, seq_len, hidden_size)}
        last_hidden_state = self.get_last_hidden_state(outputs.hidden_states[-1], full_attention_mask) # shape (num_envs, hidden_size)
        value = self.critic(last_hidden_state)

        entropy = Categorical(logits=logits).entropy()  # (num_envs, seq_len)
        masked_entropy = entropy * final_mask
        summed_entropy = masked_entropy.sum(dim=-1)

        if action_ids is None:
            return summed_log_probs, value, full_ids, full_attention_mask, prompt_lens, generated_texts
        else:
            return summed_log_probs, value, summed_entropy, final_mask

    def get_last_hidden_state(self, hidden_states, attention_mask):
        """Calculates the hidden state of the last non-padding token."""
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden_state = hidden_states[batch_indices, last_token_indices, :]
        return last_hidden_state
    
    def set_target_modules(self):
        target_modules = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        
        target_modules += [
            "attn.qkv",
            "attn.proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        self.target_modules = target_modules
        return self.target_modules