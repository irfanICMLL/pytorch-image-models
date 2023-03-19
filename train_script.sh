CUDA_VISIBLE_DEVICES=2 python train.py ./data/imagenet --model vit_tiny_patch16_384 --pretrained --sched poly --epochs 300 --warmup-epochs 5  --remode pixel --batch-size 256 --amp -j 4
