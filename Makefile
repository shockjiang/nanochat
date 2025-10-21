dev = $(shell nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -8 | awk -F ',' '{print $$1}' | paste -sd,)
host = $(shell hostname)
today =  $(shell date +%m%d)
time = $(shell date +%m%d%H%M)
GPU_NUM = 2

export CUDA_HOME=/comp_robot/shock/share/pkgs/cuda-12.1

export TOKENIZERS_PARALLELISM=True
export HF_ENDPOINT=https://hf-mirror.com
# export OMP_NUM_THREADS=64
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=64
# export NUMEXPR_MAX_THREADS=64

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

export PYTHONPATH=.
export NO_ALBUMENTATIONS_UPDATE=1
export TRITON_CACHE_DIR=/tmp/${USER}/.triton
export PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning


export NANOCHAT_BASE_DIR=${HOME}/.cache/nanochat

train:
	CUDA_VISIBLE_DEVICES=${dev} torchrun --standalone --nproc_per_node=${GPU_NUM} -m scripts.base_train -- --depth=20 --run=dummy

tok:
	python -m scripts.tok_train --max_chars=2000000000
	
deps:
	pydeps . --max-bacon=2 --show-deps --noshow