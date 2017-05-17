#!/bin/bash
# This script trains a deepspeech model in tensorflow with sorta-grad.
# usage ./train.sh  or  ./train.sh dummy


clear
cur_dir=$(cd "$(dirname $0)";pwd)
echo ${cur_dir}
export PYTHONPATH=${cur_path}:/home/matrix/inteltf/:$PYTHONPATH
# echo $PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

# activate Intel Python
# source /opt/intel/intelpython2/bin/activate

# echo "Training on utterances in order sorted by length"
#export CUDA_VISIBLE_DEVICES=0,1
# filename='../models/librispeech/train'
# datadir='../data/LibriSpeech/processed/'
# python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 280 --num_rnn_layers 7 --num_hidden 1760 --rnn_type 'bi-dir' --num_filters 32 --initial_lr 1e-4 --train_dir $filename --data_dir $datadir --use_fp32

# clear
echo "-----------------------------------"
if [ $# -eq 0 ];then
	echo "Training now on shuffled utterances"
	filename='../models/librispeech/train'
	datadir='../data/LibriSpeech/processed/'
	python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --rnn_type 'bi-dir' --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --train_dir $filename --data_dir $datadir --num_gpus 1
else
	echo "Training now on dummy data"
	python deepSpeech_train.py --batch_size 32 --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --rnn_type 'bi-dir' --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --num_gpus 1 
fi

# deactivate Intel Python
# source /opt/intel/intelpython2/bin/deactivate

# clear
# echo "-----------------------------------"
# echo "Training now on dummy data"
# filename='../models/dummy/train'
# python deepSpeech_train.py --batch_size 32 --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --checkpoint ../models/dummy --train_dir $filename



