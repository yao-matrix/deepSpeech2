#!/bin/bash
# This script trains a deepspeech model in tensorflow with sorta-grad.

clear				
echo "Training on utterances in order sorted by length"
export CUDA_VISIBLE_DEVICES=0,1
python deepSpeech_train.py --batch_size 32 --temporal_stride 3 --no-shuffle  --max_steps 280 --num_rnn_layers 3 --num_hidden 1024 --rnn_type 'bi-dir' --num_filters 128 --initial_lr 1e-4 --train_dir ../models/librispeech_1 --use_fp32 
clear
echo "-----------------------------------"
echo "Training now on shuffled utterances"
filename='../models/librispeech/train'
python deepSpeech_train.py --batch_size 32 --shuffle --max_steps 40000 --checkpoint ../models/librispeech_1 --train_dir $filename



