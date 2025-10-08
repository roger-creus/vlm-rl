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
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    
    log_every: int = 1
    """the frequency of logging interactions to a file"""
    log_dir: str = "interaction_logs"
    """the directory to save interaction logs"""

    # VLM specific arguments
    vlm_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    """The model ID from Hugging Face."""
    prompt_path: str = "prompt.txt"
    """Path to a file containing the text prompt for the VLM."""
    kl_coef: float = 0.02
    """Coefficient for the KL-divergence penalty in the reward, to stabilize training."""
    cot_lambda: float = 0.5
    """Scaling factor to down-weight the CoT reasoning tokens in log-prob calculation, as per Section 4.3."""
    max_new_tokens: int = 128
    """Maximum number of new tokens for the VLM to generate."""
    enable_compile: bool = False
    """wether to compile VLM"""
    
    use_lora: bool = False
    lora_rank: int = 1
    lora_alpha: float = 0.1
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
