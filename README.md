module load conda
mkdir -p $SCRATCH/conda_envs
export CONDA_ENVS_PATH=$SCRATCH/conda_envs
conda create -n cleanrl-vlm python=3.10 -y
conda activate $CONDA_ENVS_PATH/cleanrl-vlm
pip install -r requirements.txt
pip install vizdoom
pip install git+https://github.com/huggingface/transformers
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 
pip install flash-attn==2.7.4.post1 --no-build-isolation
conda env config vars set HF_HOME=$SCRATCH/hub
conda deactivate
conda activate $CONDA_ENVS_PATH/cleanrl-vlm

# test this in 4-gpu node
accelerate launch --config_file=deepspeed_zero2.yaml src/train_decoupled_actor_critic_cot.py  --vlm_name="Qwen/Qwen3-VL-4B-Instruct" --track --num_envs=2 --num_steps=8 --num_minibatches=8 --gradient_accumulation_steps=4 --critic_warmup_iterations=0