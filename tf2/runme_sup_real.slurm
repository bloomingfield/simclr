#!/bin/bash
#SBATCH --partition shortgpgpu
#SBATCH --qos=gpgpuresplat

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --time=00:59:59
#SBATCH --output runme_sup.out
#SBATCH --open-mode=truncate
#SBATCH --job-name=runme
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathaniel.bloomfield@unimelb.edu.au

module load fosscuda/2020b
module load tensorflow/2.6.0-python-3.8.6

source ~/venvs/tf_gpu/bin/activate

python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=-1 --zero_init_logits_layer=True \
  --optimizer=lars --learning_rate=0.005 \
  --learning_rate_scaling=sqrt --weight_decay=0 --image_size=32  --resnet_depth=18  --global_bn=False \
  --train_steps=1 --train_batch_size=1024 --warmup_epochs=0 \
  --dataset=cifar10 --eval_split=test \
  --data_dir=data/cifar10 --model_dir=cifar10_models/finetune12 --checkpoint=cifar10_models/firsttry_real/ckpt-585.index \
  --use_tpu=False  \
  --num_proj_layers=2 --ft_proj_selector=0
