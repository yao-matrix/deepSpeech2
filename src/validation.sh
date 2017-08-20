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

# clear
echo "-----------------------------------"
echo "Start testing"

nchw=true
engine="mkl" # tf, mkl, cudnn_rnn, mkldnn_rnn

if [[ !${nchw} && (${engine} == "tf" || ${engine} == "cudnn_rnn") ]];then
	python deepSpeech_test.py --eval_data 'val' --nchw ${nchw} --engine ${engine}
elif [[ ${nchw} && (${engine} == "mkl" || ${engine} == "mkldnn_rnn") ]];then
	python deepSpeech_test.py --eval_data 'val' --nchw ${nchw} --engine ${engine}
else
	echo "unsupported parameter combination"
fi
echo "Done"

# deactivate Intel Python
# source /opt/intel/intelpython2/bin/deactivate

