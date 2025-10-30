import torch
import torch.nn as nn
import numpy as np
from peft import LoraConfig, get_peft_model

from IPython import embed
from torch.distributions.categorical import Categorical
from transformers import AutoTokenizer, AutoProcessor
from src.utils.utils import numpy_to_pil, gc_cuda_cleanup, get_model_class

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def default_target_modules():
    """Provides a default list of target modules for LoRA."""
    target_modules = [
        # --- Text Model Modules ---
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        
        # --- Vision Model Modules ---
        "attn.qkv",
        "attn.proj",
        "mlp.linear_fc1",
        "mlp.linear_fc2",

        # --- Language Model Head ---
        "lm_head",

        # --- Vision Merger Layers ---
        "merger.linear_fc1",
        "merger.linear_fc2",
    ]
    # Remove duplicates while preserving order
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
            min_pixels = 768 * 32 * 32,
            max_pixels = 768 * 32 * 32,
        )
        mdl_cls = get_model_class(vlm_name)
        self.model = mdl_cls.from_pretrained(
            vlm_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        
        # cast the lm_head to float32
        #self.model.lm_head = self.model.lm_head.to(torch.float32)

    def preprocess_obs_and_text(self, obs, text_prompts):
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
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden_state = hidden_states[batch_indices, last_token_indices, :]
        return last_hidden_state

    def get_trainable_params(self):
        """Returns a list of parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

# --- CRITIC HEAD ---
class CriticHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def forward(self, hidden):
        return self.net(hidden)

class DecoupledActorCriticVLM_COT(nn.Module):
    """
    Decoupled actor-critic with a SINGLE VLM and two LoRA adapters.
    """
    def __init__(
        self,
        vlm_name: str,
        max_new_tokens: int = 128,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_lora = use_lora
        self.vlm = BaseVLM(vlm_name, max_new_tokens)
        
        hidden_size = self.vlm.model.config.text_config.hidden_size
        self.critic_head = CriticHead(hidden_size).to(self.vlm.model.dtype)
        self.max_new_tokens = max_new_tokens

        if self.use_lora:
            actor_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=default_target_modules(),
                bias="none",
            )
            
            critic_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=default_target_modules(),
                bias="none",
            )

            self.vlm.model = get_peft_model(
                self.vlm.model, 
                actor_lora_config, 
                adapter_name="actor"
            )

            self.vlm.model.add_adapter(
                "critic",
                critic_lora_config
            )
            
    def get_trainable_params(self):
        params = self.vlm.get_trainable_params()
        params.extend(list(self.critic_head.parameters()))
        return params

    def get_action(self, obs=None, text_prompts=None, action_ids=None, prompt_lens=None):
        if self.use_lora:
            # --- Set the active adapter to 'actor' ---
            self.vlm.model.set_adapter("actor")
        
        batch_size = len(text_prompts)
        inputs = self.vlm.preprocess_obs_and_text(obs, text_prompts)
        pixel_values = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw

        if action_ids is None:
            full_ids = self.vlm.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
            )
            generated_texts = self.vlm.processor.batch_decode(
                full_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            prompt_lens = torch.tensor([inputs.input_ids.shape[1]] * batch_size, device=self.vlm.model.device)
        else:
            full_ids = action_ids

        attention_mask = (full_ids != self.vlm.processor.tokenizer.pad_token_id).long()
        outputs = self.vlm.model(
            input_ids=full_ids,
            image_grid_thw=image_grid_thw,
            pixel_values=pixel_values,
            output_hidden_states=True,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        log_probs_all = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = full_ids[:, 1:]
        log_probs = torch.gather(log_probs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)

        entropy = Categorical(logits=logits[:, :-1, :]).entropy()

        if action_ids is None:
            return log_probs, full_ids, prompt_lens, generated_texts
        else:
            return log_probs, entropy

    def get_value(self, obs, prompt_text):
        if self.use_lora:
            # --- Set the active adapter to 'critic' ---
            self.vlm.model.set_adapter("critic")
        
        inputs = self.vlm.preprocess_obs_and_text(obs, prompt_text)
        outputs = self.vlm.model(
            **inputs,
            output_hidden_states=True,
        )
        last_hidden = self.vlm.last_hidden_state(outputs.hidden_states[-1], inputs['attention_mask'])
        return self.critic_head(last_hidden)
    
    
class DecoupledActorCriticVLM_Action(nn.Module):
    def __init__(
        self,
        vlm_name: str,
        available_actions,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.use_lora = use_lora
        self.vlm = BaseVLM(vlm_name)
        
        hidden_size = self.vlm.model.config.text_config.hidden_size
        self.critic_head = CriticHead(hidden_size).to(device=device, dtype=self.vlm.model.dtype)
        
        self.available_actions = available_actions
        self.num_actions = len(available_actions)

        # --- Tokenize and store available actions ---
        tokenized_actions = self.vlm.processor.tokenizer(
            self.available_actions, 
            padding=True, 
            return_tensors="pt",
            add_special_tokens=False
        )
        
        self.action_ids = tokenized_actions.input_ids.to(device)
        self.action_mask = tokenized_actions.attention_mask.to(device)
        self.action_lens = self.action_mask.sum(dim=1)

        if self.use_lora:
            actor_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=default_target_modules(),
                bias="none",
            )
            
            critic_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=default_target_modules(),
                bias="none",
            )

            self.vlm.model = get_peft_model(
                self.vlm.model, 
                actor_lora_config, 
                adapter_name="actor"
            )

            self.vlm.model.add_adapter(
                "critic",
                critic_lora_config
            )
            
    def get_trainable_params(self):
        params = self.vlm.get_trainable_params()
        params.extend(list(self.critic_head.parameters()))
        return params

    def _get_action_scores(self, obs, text_prompts):
        if self.use_lora:
            self.vlm.model.set_adapter("actor")
            
        base_inputs = self.vlm.preprocess_obs_and_text(obs, text_prompts)
        prompt_ids = base_inputs.input_ids
        prompt_mask = base_inputs.attention_mask
        
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        
        # 2. Expand inputs to evaluate all actions for each batch item
        # [B, N_act, S_prompt]
        expanded_prompt_ids = prompt_ids.unsqueeze(1).repeat(1, self.num_actions, 1)
        expanded_prompt_mask = prompt_mask.unsqueeze(1).repeat(1, self.num_actions, 1)
        assert expanded_prompt_ids.shape == (batch_size, self.num_actions, prompt_len)
        
        # [B, N_act, S_act]
        expanded_action_ids = self.action_ids.unsqueeze(0).repeat(batch_size, 1, 1)
        expanded_action_mask = self.action_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        assert expanded_action_ids.shape == (batch_size, self.num_actions, self.action_ids.shape[1])
        assert expanded_action_mask.shape == (batch_size, self.num_actions, self.action_mask.shape[1])
        
        # 3. Concatenate prompts and actions
        # [B, N_act, S_prompt + S_act]
        combined_ids = torch.cat([expanded_prompt_ids, expanded_action_ids], dim=2)
        combined_mask = torch.cat([expanded_prompt_mask, expanded_action_mask], dim=2)
        assert combined_ids.shape == (batch_size, self.num_actions, prompt_len + self.action_ids.shape[1])
        assert combined_mask.shape == (batch_size, self.num_actions, prompt_len + self.action_mask.shape[1])
        
        # 4. Flatten for a single batch forward pass
        # [B * N_act, S_prompt + S_act]
        flat_ids = combined_ids.view(batch_size * self.num_actions, -1)
        flat_mask = combined_mask.view(batch_size * self.num_actions, -1)
        assert flat_ids.shape == (batch_size * self.num_actions, prompt_len + self.action_ids.shape[1])
        assert flat_mask.shape == (batch_size * self.num_actions, prompt_len + self.action_mask.shape[1])
        
        # 5. Expand vision features
        # [B * num_patches, dim]
        pixel_values = base_inputs.pixel_values 
        num_patches = pixel_values.shape[0] // batch_size
        dim = pixel_values.shape[1]
        
        pixel_values_reshaped = pixel_values.view(batch_size, num_patches, dim)
        # [B, 1, num_patches, dim] -> [B, N_act, num_patches, dim]
        expanded_pixel_values = pixel_values_reshaped.unsqueeze(1).repeat(
            1, self.num_actions, 1, 1
        )
        
        # [B * N_act, num_patches, dim] -> [B * N_act * num_patches, dim]
        flat_pixel_values = expanded_pixel_values.contiguous().view(
            batch_size * self.num_actions * num_patches, dim
        )
        assert flat_pixel_values.shape == (batch_size * self.num_actions * num_patches, dim)
        
        # [B, S_grid] -> [B * N_act, S_grid]
        image_grid_thw = base_inputs.image_grid_thw
        expanded_image_grid = image_grid_thw.unsqueeze(1).repeat(
            1, self.num_actions, *([1] * (image_grid_thw.dim() - 1))
        ).view(batch_size * self.num_actions, *image_grid_thw.shape[1:])
        assert expanded_image_grid.shape == (batch_size * self.num_actions, *image_grid_thw.shape[1:])
        
        # 6. Run forward pass
        outputs = self.vlm.model(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            pixel_values=expanded_pixel_values,
            image_grid_thw=expanded_image_grid,
            output_hidden_states=False,
        )
        # [B * N_act, S_comb, V]
        logits = outputs.logits
        assert logits.shape == (batch_size * self.num_actions, prompt_len + self.action_ids.shape[1], self.vlm.model.config.text_config.vocab_size)
        
        # 7. Calculate log-probabilities for the action tokens
        # We need logits from index (prompt_len - 1) up to (end - 1)
        # to predict tokens from index prompt_len to end
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get the logits corresponding to the *action* part
        # [B * N_act, S_act, Vocab]
        action_logits = log_probs_all[:, prompt_len - 1 : -1, :]
        
        # Get the target action token IDs
        # [B * N_act, S_act]
        action_ids_for_gather = flat_ids[:, prompt_len:]
        
        # Gather the log-probabilities of the target tokens
        # [B * N_act, S_act]
        log_probs_gathered = torch.gather(
            action_logits, 2, action_ids_for_gather.unsqueeze(-1)
        ).squeeze(-1)
        
        # 8. Mask out padding tokens within the actions
        # [B * N_act, S_act]
        flat_action_mask = expanded_action_mask.view(batch_size * self.num_actions, -1)
        log_probs_masked = log_probs_gathered * flat_action_mask
        
        # 9. Sum log-probabilities to get score for each full action
        # [B * N_act]
        action_total_logprobs = log_probs_masked.sum(dim=1)
        
        # 10. Reshape to [B, N_act]
        action_scores = action_total_logprobs.view(batch_size, self.num_actions)
        return action_scores

    def get_action(self, obs=None, text_prompts=None):
        # [B, N_act]
        action_scores = self._get_action_scores(obs, text_prompts)
        
        dist = Categorical(logits=action_scores)
        sampled_action_indices = dist.sample() # [B]
        
        sampled_log_prob = dist.log_prob(sampled_action_indices) # [B]
        
        batch_size = action_scores.shape[0]
        sampled_actions_str = [self.available_actions[i] for i in sampled_action_indices]
        sampled_action_ids = self.action_ids[sampled_action_indices] # [B, S_act]
        sampled_action_mask = self.action_mask[sampled_action_indices] # [B, S_act]
        
        return (
            sampled_action_indices, # [B]
            sampled_log_prob,       # [B]
            sampled_actions_str,    # List[str] of len B
            sampled_action_ids,     # [B, S_act]
            sampled_action_mask,    # [B, S_act]
        )

    def get_actor_outputs(self, obs, text_prompts, taken_action_indices):
        # [B, N_act]
        action_scores = self._get_action_scores(obs, text_prompts)
        
        # Form distribution
        dist = Categorical(logits=action_scores)
        
        # Get log-prob of the actions that were *actually taken*
        # [B]
        log_prob = dist.log_prob(taken_action_indices.long())
        
        # Get entropy of the distribution
        # [B]
        entropy = dist.entropy()
        
        return log_prob, entropy


    def get_value(self, obs, prompt_text):
        """
        Gets the value (critic) estimate for the current state.
        """
        if self.use_lora:
            # --- Set the active adapter to 'critic' ---
            self.vlm.model.set_adapter("critic")
        
        inputs = self.vlm.preprocess_obs_and_text(obs, prompt_text)
        outputs = self.vlm.model(
            **inputs,
            output_hidden_states=True,
        )
        # Use the last non-padded token's hidden state
        last_hidden = self.vlm.last_hidden_state(
            outputs.hidden_states[-1], inputs['attention_mask']
        )
        return self.critic_head(last_hidden)