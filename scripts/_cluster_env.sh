#!/usr/bin/env bash
# Shared cluster env setup. Source before launching training.
#
#   source scripts/_cluster_env.sh

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_HOME="${HF_HOME:-$SCRATCH/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Inv-11 determinism knobs default on; opt out by unsetting before sourcing.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
