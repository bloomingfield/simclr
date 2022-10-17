#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --qos=gpgpuresplat

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --time=10:59:59
#SBATCH --output runme.out
#SBATCH --open-mode=truncate
#SBATCH --job-name=runme
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathaniel.bloomfield@unimelb.edu.au

module load fosscuda/2020b
module load tensorflow/2.6.0-python-3.8.6

python run.py --train_mode=pretrain  --train_batch_size=256 --train_steps=195  \
--learning_rate=0.2 --learning_rate_scaling=sqrt --proj_out_dim=64 --num_proj_layers=1 \
--weight_decay=0.0 --temperature=0.2   --dataset=cifar10 --data_dir=data/cifar10 \
--image_size=32 --eval_split=test --resnet_depth=18   --use_blur=False --color_jitter_strength=0.5  \
--model_dir=cifar10_models/firsttry_aug2_8 --use_tpu=False --cache_dataset=True --warmup_epochs=0 \
--optimizer=lars
