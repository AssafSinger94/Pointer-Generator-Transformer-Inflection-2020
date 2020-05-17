#!/bin/bash
gpu=0
data_dir=task0-data/out
aug_dir=task0-data/aug
vocab_dir=task0-vocab
ckpt_dir=checkpoints
pred_dir=task0-pred
lang=$1
arch=pointer-generator
resume=False

epochs=1000
#epochs=20
eval_every=1
bs=400
lr=0.001
beta2=0.98
scheduler=warmupinvsqr
patience=$2
min_lr=1e-5
warmup=4000

model_copy=$3

# transformer
num_layers=4
fcn_dim=1024
embed_dim=256
num_heads=4
dropout=${4:-0.3}

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --train $aug_dir/$lang.aug \
    --dev $data_dir/$lang.dev \
    --vocab $vocab_dir/$lang-vocab \
    --checkpoints-dir $ckpt_dir/$arch/$lang/$scheduler/$model_copy \
    --resume $resume \
    --epochs $epochs --batch-size $bs --eval-every $eval_every \
    --arch $arch --embed-dim $embed_dim --fcn-dim $fcn_dim \
    --num-layers $num_layers --num-heads $num_heads --dropout $dropout \
    --lr $lr --beta2 $beta2 \
    --scheduler $scheduler --patience $patience --min-lr $min_lr --warmup-steps $warmup

CUDA_VISIBLE_DEVICES=$gpu python generate.py \
    --model-checkpoint $ckpt_dir/$arch/$lang/$scheduler/$model_copy/model_best.pth \
    --arch $arch --embed-dim $embed_dim --fcn-dim $fcn_dim \
    --num-layers $num_layers --num-heads $num_heads --dropout $dropout \
    --test $data_dir/$lang.covered-dev \
    --vocab $vocab_dir/$lang-vocab \
    --pred $pred_dir/$arch/$scheduler/$lang-$model_copy.pred

python evaluate.py \
    --target $data_dir/$lang.dev \
    --pred $pred_dir/$arch/$scheduler/$lang-$model_copy.pred