from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = "vlm_ppo_finetuning"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    enable_compile: bool = False
    """whether to compile VLM"""
    
    # --- Logging ---
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL-VLM"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_id: str = None
    """the id of the wandb run to resume"""
    log_every: int = 1
    """the frequency of logging interactions to a file"""
    log_dir: str = "interaction_logs"
    """the directory to save interaction logs"""
    checkpoint_dir: str = ""
    """the directory to save checkpoints"""
    checkpoint_interval: int = 10
    """the interval to save checkpoints"""

    # --- Config ---
    env_id: str = "VizdoomCorridor-v0"
    """the id of the environment"""
    total_timesteps: int = 500_000
    """total timesteps of the experiments"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 128
    """the number of mini-batches"""
    gradient_accumulation_steps: int = 32
    """the number of gradient accumulation steps"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    dual_clip_c: float = 0.0
    """the coefficient for the Dual-Clip PPO"""
    logratio_clamp: float = 20.0
    """the clamp value for the log-ratio"""
    clip_coef_lower: float = 0.2
    """the lower clip coefficient for the Dual-Clip PPO"""
    clip_coef_upper: float = 0.2
    """the upper clip coefficient for the Dual-Clip PPO"""
    
    # --- Optimizer ---
    learning_rate: float = 5e-5
    """the learning rate of the optimizer"""
    weight_decay: float = 0.0
    """the weight decay of the optimizer"""
    lr_warmup_fraction: float = 0.1
    """the fraction of the total number of iterations to warm up the learning rate"""
    reward_scale: float = 0.01
    """the scale of the reward"""

    # --- VLM specific arguments ---
    vlm_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    """The model ID from Hugging Face."""
    prompt_actor_path: str = "prompts/corridor/actor.txt"
    """Path to a file containing the text prompt for the VLM."""
    prompt_critic_path: str = "prompts/corridor/critic.txt"
    """Path to a file containing the text prompt for the VLM."""
    max_new_tokens: int = 512
    """Maximum number of new tokens for the VLM to generate."""
    max_seq_len: int = 3072
    """Maximum sequence length for the VLM to generate. Longer sequences will be truncated."""
    critic_warmup_iterations: int = 5
    """Number of iterations to warm up the critic."""
    warmup_epochs: int = 5
    """Number of epochs to warm up the critic."""
    
    # --- LoRa ---
    lora_rank: int = 32
    """the rank of the LoRA adapters"""
    lora_alpha: float = 64
    """the alpha of the LoRA adapters"""
    
    # will be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
