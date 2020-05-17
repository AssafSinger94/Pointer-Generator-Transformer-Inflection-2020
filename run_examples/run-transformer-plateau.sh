#!/bin/bash
gpu=0
data_dir=task0-data
vocab_dir=task0-vocab
ckpt_dir=checkpoints
pred_dir=task0-pred
lang=$1
arch=transformer

epochs=300
eval_every=7
bs=400
lr=0.001
beta2=0.98
scheduler=ReduceLROnPlateau
min_lr=1e-5
warmup=4000

# transformer
num_layers=4
fcn_dim=1024
embed_dim=256
num_heads=4
dropout=${2:-0.3}

for i in 1 2 3; do
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --train $data_dir/$lang.trn \
    --dev $data_dir/$lang.dev \
    --vocab $vocab_dir/$lang-vocab \
    --checkpoints-folder $ckpt_dir/$arch/$lang/plt \
    --arch $arch --epochs $epochs --batch-size $bs --eval-every $eval_every \
    --embed-dim $embed_dim --fcn-dim $fcn_dim --dropout $dropout \
    --num-layers $num_layers --num-heads $num_heads \
    --lr $lr --beta2 $beta2 \
    --scheduler $scheduler --min-lr $min_lr --warmup-steps $warmup

python generate.py \
    --model-checkpoint $ckpt_dir/$arch/$lang/plt/model_best.pth \
    --test $data_dir/$lang.covered-dev \
    --vocab $vocab_dir/$lang-vocab \
    --pred $pred_dir/$arch/$lang-plt.pred

python evaluate.py \
    --target $data_dir/$lang.dev \
    --pred $pred_dir/$arch/$lang-plt.pred
done
