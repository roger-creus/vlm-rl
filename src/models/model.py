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
            #min_pixels = 210 * 160 * 3,
            #max_pixels = 210 * 160 * 3,
            #patch_size=7,
        )
        mdl_cls = get_model_class(vlm_name)
        self.model = mdl_cls.from_pretrained(
            vlm_name,
            dtype=torch.bfloat16,
            #dtype=torch.float32,
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
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def forward(self, hidden):
        return self.net(hidden)

class DecoupledActorCriticVLM(nn.Module):
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
            
            # lm_head = self.vlm.model.base_model.model.lm_head
            # for lora_param_name, lora_param in lm_head.lora_A.items():
            #     lora_param.weight.data = lora_param.weight.data.to(torch.float32)
            # for lora_param_name, lora_param in lm_head.lora_B.items():
            #     lora_param.weight.data = lora_param.weight.data.to(torch.float32)
            
            # --- VERIFICATION ---
            print("--- Verifying Model Dtypes ---")

            # 1. Check a "normal" model LoRA layer
            try:
                lora_weight = self.vlm.model.base_model.model.model.language_model.layers[0].self_attn.q_proj.lora_A.actor.weight
                print(f"LoRA Weight (q_proj):    {lora_weight.dtype}")
            except Exception as e:
                print(f"Could not check q_proj LoRA weight: {e}")

            # 2. Check that layer's Base Weight
            try:
                base_weight = self.vlm.model.base_model.model.model.language_model.layers[0].self_attn.q_proj.weight
                print(f"Base Layer (q_proj):     {base_weight.dtype}")
            except Exception as e:
                print(f"Could not check q_proj base weight: {e}")

            # 3. Check the LM Head Base Weight
            try:
                lm_head_weight = self.vlm.model.base_model.model.lm_head.weight
                print(f"Base Layer (lm_head):    {lm_head_weight.dtype}")
            except Exception as e:
                print(f"Could not check LM Head base weight: {e}")
            
            # 4. Check the "lm_head" LoRA layer
            try:
                lm_head_lora_weight = self.vlm.model.base_model.model.lm_head.lora_A.actor.weight
                print(f"LoRA Weight (lm_head):   {lm_head_lora_weight.dtype}")
            except Exception as e:
                print(f"Could not check LM Head LoRA weight: {e}")

            # 5. Check Critic Head
            try:
                critic_head_weight = self.critic_head.net[0].weight
                print(f"Critic Head Weight:      {critic_head_weight.dtype}")
            except Exception as e:
                print(f"Could not check Critic Head weight: {e}")
            
            print("---------------------------------")
            
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
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs_all, 2, full_ids.unsqueeze(-1)).squeeze(-1)

        entropy = Categorical(logits=logits).entropy()

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