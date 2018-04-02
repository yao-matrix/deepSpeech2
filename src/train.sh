#!/bin/bash

clear
cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
export PYTHONPATH=${cur_path}:$PYTHONPATH
# echo $PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

# activate Intel Python
# source /opt/intel/intelpython2/bin/activate

# environment variables
unset TF_CPP_MIN_VLOG_LEVEL
# export TF_CPP_MIN_VLOG_LEVEL=2

# clear
echo "-----------------------------------"
echo "Start training"

dummy=False   # True or False
nchw=False    # True or False
debug=False   # True or False
engine="tf"   # tf, mkl, cudnn_rnn, mkldnn_rnn

# echo $dummy

config_check_one=`test "${nchw}" = "False" && test "${engine}"x = "tf"x -o "${engine}"x = "cudnn_rnn"x && echo 'OK'`
# echo "check one: "$config_check_one
config_check_two=`test "${nchw}" = "True" && test "${engine}"x == "mkl"x -o "${engine}"x = "mkldnn_rnn"x && echo 'OK'`
# echo "check two: "$config_check_two
check=`test ${config_check_one}x = "OK"x -o ${config_check_two}x = "OK"x && echo 'OK'`
# echo "check: "$check

if [[ ${check}x != "OK"x ]];then
    echo "unsupported configuration conbimation"
    exit -1
fi

model_dir='../models/librispeech/train'
data_dir='/home/matrix/data/processed/'
python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 400000 --num_rnn_layers 3 --num_hidden 1024 --num_filters 32 --initial_lr 5e-4 --train_dir $model_dir --data_dir $data_dir --debug ${debug} --nchw ${nchw} --engine ${engine} --dummy ${dummy}

echo "Done"

# deactivate Intel Python
# source /opt/intel/intelpython2/bin/deactivate

