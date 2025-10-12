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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL-VLM"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MsPacmanNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-5
    """the learning rate of the optimizer"""
    weight_decay: float = 0.01
    """the weight decay of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 128
    """the number of mini-batches"""
    gradient_accumulation_steps: int = 32
    """the number of gradient accumulation steps"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    
    log_every: int = 1
    """the frequency of logging interactions to a file"""
    log_dir: str = "interaction_logs"
    """the directory to save interaction logs"""

    # VLM specific arguments
    vlm_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    """The model ID from Hugging Face."""
    prompt_actor_path: str = "prompt_actor.txt"
    """Path to a file containing the text prompt for the VLM."""
    prompt_critic_path: str = "prompt_critic.txt"
    """Path to a file containing the text prompt for the VLM."""
    max_new_tokens: int = 128
    """Maximum number of new tokens for the VLM to generate."""
    max_seq_len: int = 1024
    """Maximum sequence length for the VLM to generate. Longer sequences will be truncated."""
    critic_warmup_iterations: int = 10
    """Number of iterations to warm up the critic."""

    enable_compile: bool = False
    """wether to compile VLM"""
    
    # LoRa specific arguments
    lora_rank: int = 32
    """the rank of the LoRA adapters"""
    lora_alpha: float = 64
    """the alpha of the LoRA adapters"""
    
    # Dual-Clip PPO specific arguments
    dual_clip_c: float = 3.0
    """the coefficient for the Dual-Clip PPO"""
    logratio_clamp: float = 20.0
    """the clamp value for the log-ratio"""
    clip_coef_lower: float = 0.2
    """the lower clip coefficient for the Dual-Clip PPO"""
    clip_coef_upper: float = 0.2
    """the upper clip coefficient for the Dual-Clip PPO"""
    
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
