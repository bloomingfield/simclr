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

python run.py --mode=train_then_eval --train_mode=finetune --train_batch_size=512 --train_epochs=100 \
--fine_tune_after_block=-1 --zero_init_logits_layer=True --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' --global_bn=False \
--optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 --warmup_epochs=0 \
--dataset=cifar10 --data_dir=data/cifar10 \
--image_size=32 --eval_split=test --resnet_depth=18 \
--checkpoint=cifar10_models/firsttry_real \
--model_dir=cifar10_models/finetune2  --use_tpu=False
