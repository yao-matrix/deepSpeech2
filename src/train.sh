#!/bin/bash
# This script trains a deepspeech model in tensorflow with sorta-grad.
# usage ./train.sh  or  ./train.sh dummy


clear
cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
export PYTHONPATH=${cur_path}:/home/matrix/inteltf/:$PYTHONPATH
# echo $PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

# activate Intel Python
# source /opt/intel/intelpython2/bin/activate

# environment variables
unset TF_CPP_MIN_VLOG_LEVEL
# export TF_CPP_MIN_VLOG_LEVEL=1

# echo "Training on utterances in order sorted by length"
# export CUDA_VISIBLE_DEVICES=0,1
# filename='../models/librispeech/train'
# datadir='../data/LibriSpeech/processed/'
# python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 280 --num_rnn_layers 7 --num_hidden 1760 --rnn_type 'bi-dir' --num_filters 32 --initial_lr 1e-4 --train_dir $filename --data_dir $datadir --use_fp32

# clear
echo "-----------------------------------"
echo "Start training"

dummy=0  # 0, 1
nchw=1   # 0, 1
debug=0  # 0, 1
engine="mkl" # tf, mkl, cudnn_rnn, mkldnn_rnn

config_check_one=`test ${nchw} -eq 0 && test "${engine}"x = "tf"x -o "${engine}"x = "cudnn_rnn"x && echo 'OK'`
# echo "check one: "$config_check_one
config_check_two=`test ${nchw} -eq 1 && test "${engine}"x == "mkl"x -o "${engine}"x = "mkldnn_rnn"x && echo 'OK'`
# echo "check two: "$config_check_two
check=`test ${config_check_one}x = "OK"x -o ${config_check_two}x = "OK"x && echo 'OK'`
# echo "check: "$check

if [[ ${check}x != "OK"x ]];then
    echo "unsupported configuration conbimation"
    exit -1
fi

filename='../models/librispeech/train'
datadir='../data/LibriSpeech/processed/'
python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --train_dir $filename --data_dir $datadir --debug ${debug} --nchw ${nchw} --engine ${engine} --dummy ${dummy}

echo "Done"

# deactivate Intel Python
# source /opt/intel/intelpython2/bin/deactivate

