# README

## environment

```sh
mamba create -n iterative
mamba install python=3.11 pytorch pytorch-cuda torchvision opencv -c pytorch -c nvidia
mamba install pkg-config
pip install ffcv
pip install lightning diffusers torchmetrics einops
pip install rootutils rich
pip install omegaconf hydra-core hydra-colorlog hydra-optuna-sweeper # configs
pip install tensorboard torch-tb-profiler mlflow # logging
```

## checkpoint

1. install the git-lfs

```sh
sudo dnf install -y git-lfs
```

1. download the checkpoint from mirror

```sh
mkdir hf_pipelines
cd hf_pipelines && git clone https://hf-mirror.com/google/ddpm-cifar10-32
# Directly download the ckpt file from huggingface is too slow
```

download via lfs

```sh
cd ddpm-cifar10-32
git lfs install
git lfs pull
```

## run

```sh
python src/train.py # append various options in configs/ here
```
