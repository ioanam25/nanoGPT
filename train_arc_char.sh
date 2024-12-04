#!/bin/bash
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0015
#SBATCH --mem=160Gb
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=4

#SBATCH --nodes=1
#SBATCH --gres=gpu:4

#SBATCH --job-name=size2_source_0_no_thinking
#SBATCH --output slurm/size2_source_0_no_thinking.out

torchrun --standalone --nproc_per_node=4 train.py config/train_arc_char.py --gradient_accumulation_steps=4 --batch_size=8 --no_thinking=1 --curr_source=0 --out_dir='no_thinking/source_0/size2/1' --n_layer=12 --n_head=8 --n_embd=768 --wandb_project='0_size2' --wandb_run_name='one'
torchrun --standalone --nproc_per_node=4 train.py config/train_arc_char.py --gradient_accumulation_steps=4 --batch_size=8 --no_thinking=1 --curr_source=0 --out_dir='no_thinking/source_0/size2/2' --n_layer=12 --n_head=8 --n_embd=768 --wandb_project='0_size2' --wandb_run_name='two'
torchrun --standalone --nproc_per_node=4 train.py config/train_arc_char.py --gradient_accumulation_steps=4 --batch_size=8 --no_thinking=1 --curr_source=0 --out_dir='no_thinking/source_0/size2/3' --n_layer=12 --n_head=8 --n_embd=768 --wandb_project='0_size2' --wandb_run_name='three'
torchrun --standalone --nproc_per_node=4 train.py config/train_arc_char.py --gradient_accumulation_steps=4 --batch_size=8 --no_thinking=1 --curr_source=0 --out_dir='no_thinking/source_0/size2/4' --n_layer=12 --n_head=8 --n_embd=768 --wandb_project='0_size2' --wandb_run_name='four'
torchrun --standalone --nproc_per_node=4 train.py config/train_arc_char.py --gradient_accumulation_steps=4 --batch_size=8 --no_thinking=1 --curr_source=0 --out_dir='no_thinking/source_0/size2/5' --n_layer=12 --n_head=8 --n_embd=768 --wandb_project='0_size2' --wandb_run_name='five'
