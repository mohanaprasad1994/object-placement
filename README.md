Tensorflow code for image placement

input images = TrainA
segmentation mask = TrainB

modified code from https://github.com/xhujoy/CycleGAN-tensorflow

To train: CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=horse2zebra

Tensorboard: tensorboard --logdir=./logs

To test: CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=horse2zebra --phase=test