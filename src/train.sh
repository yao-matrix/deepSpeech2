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

# environment variables
unset TF_CPP_MIN_VLOG_LEVEL
# export TF_CPP_MIN_VLOG_LEVEL=1

# echo "Training on utterances in order sorted by length"
#export CUDA_VISIBLE_DEVICES=0,1
# filename='../models/librispeech/train'
# datadir='../data/LibriSpeech/processed/'
# python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 280 --num_rnn_layers 7 --num_hidden 1760 --rnn_type 'bi-dir' --num_filters 32 --initial_lr 1e-4 --train_dir $filename --data_dir $datadir --use_fp32

# clear
echo "-----------------------------------"
echo "Start training"

dummy=false
nchw=true
debug=true
engine="mkl" # tf, mkl, cudnn_rnn, mkldnn_rnn

if [[ ${dummy} ]]
then
	if [[ !${nchw} && (${engine} == "tf" || ${engine} == "cudnn_rnn") ]];then
		python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --dummy ${dummy} --debug ${debug} --nchw ${nchw} --engine ${engine}
	elif [[ ${nchw} && (${engine} == "mkl" || ${engine} == "mkldnn_rnn") ]];then
		python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --dummy ${dummy} --debug ${debug} --nchw ${nchw} --engine ${engine}
	else
		echo "unsupported parameter combination"
	fi
fi

if [[ !${dummy} ]]
then
	filename='../models/librispeech/train'
	datadir='../data/LibriSpeech/processed/'
	if [[ !${nchw} && (${engine} == "tf" || ${engine} == "cudnn_rnn") ]];then
		python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --train_dir $filename --data_dir $datadir --debug ${debug} --nchw ${nchw} --engine ${engine}
	elif [[ ${nchw} && (${engine} == "mkl" || ${engine} == "mkldnn_rnn") ]];then
		python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --train_dir $filename --data_dir $datadir --debug ${debug} --nchw ${nchw} --engine ${engine}
	else
		echo "unsupported parameter combination"
	fi
fi
echo "Done"

# deactivate Intel Python
# source /opt/intel/intelpython2/bin/deactivate

