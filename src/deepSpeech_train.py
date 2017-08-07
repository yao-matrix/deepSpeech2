# Author: Lakshmi Krishnan
# Email: lkrishn7@ford.com
# Author: YAO Matrix
# Email: yaoweifeng0301@126.com


"""A script to train a deepSpeech model on LibriSpeech data using multiple GPUs
with synchronous updates (data parallel training).

References:
1. Hannun, Awni, et al. "Deep speech: Scaling up end-to-end
speech recognition." arXiv preprint arXiv:1412.5567 (2014).
2. Amodei, Dario, et al. "Deep speech 2: End-to-end
 speech recognition in english and mandarin."
 arXiv preprint arXiv:1512.02595 (2015).

Accuracy:
deepSpeech_multi_gpu_train.py achieves 15% CER on LibriSpeech data
after 30k steps (~100 epochs of data) as judged by deepSpeech_test.py.

Speed: With batch_size 128.

System             | Step Time (sec/batch)  |     Loss
--------------------------------------------------------------------
3 TitanX Pascal    | 0.25-1.45              | < 20 at 20K steps    (32 hours)


Usage:
Please see the tutorial and website for how to download the LibriSpeech
data set and train the model.

http://github.com/fordspeech/deepSpeech
"""
from datetime import datetime
import os.path
import re
import time
import argparse
import json
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline
import helper_routines
import deepSpeech_dummy
from tensorflow.python import debug as tf_debug

DEBUG = False
NCHW = False
DUMMY = True

if NCHW is True:
  import deepSpeech_NCHW as deepSpeech
else:
  import deepSpeech

from setenvs import setenvs
from setenvs import arglist
args = arglist()

profiling = []

g = tf.Graph()
if DUMMY:
    with g.as_default():
        feats_batch = tf.placeholder(dtype = tf.float32, shape = [None, None, deepSpeech_dummy.freq_bins])
        idx_batch = tf.placeholder(dtype = tf.int64)
        vals_batch = tf.placeholder(dtype = tf.int64)
        shape_batch = tf.placeholder(dtype = tf.int64)
        # labels_batch = tf.sparse_placeholder(dtype = tf.int32)
        seq_lens_batch = tf.placeholder(dtype = tf.int32, shape = [None])

def parse_args():
    " Parses command line arguments."
    num_gpus = len([x for x in device_lib.list_local_devices()
                    if x.device_type == "GPU"])
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type = str,
                        default = '../models/librispeech/train',
                        help = 'Directory to write event logs and checkpoints')
    parser.add_argument('--platform', type = str,
                        default = 'knl',
                        help = 'running platform: knl or bdw')
    parser.add_argument('--data_dir', type = str,
                        default = '',
                        help = 'Path to the audio data directory')
    parser.add_argument('--max_steps', type = int, default = 20000,
                        help = 'Number of batches to run')
    parser.add_argument('--num_gpus', type = int, default = num_gpus,
                        help = 'How many GPUs to use')
    parser.add_argument('--log_device_placement', type = bool, default = False,
                        help = 'Whether to log device placement')
    parser.add_argument('--batch_size', type = int, default = 32,
                        help = 'Number of inputs to process in a batch per GPU')
    parser.add_argument('--temporal_stride', type = int, default = 1,
                        help = 'Stride along time')

    feature_parser = parser.add_mutually_exclusive_group(required = False)
    feature_parser.add_argument('--shuffle', dest = 'shuffle',
                                action = 'store_true')
    feature_parser.add_argument('--no-shuffle', dest = 'shuffle',
                                action = 'store_false')
    parser.set_defaults(shuffle = True)

    feature_parser = parser.add_mutually_exclusive_group(required = False)
    feature_parser.add_argument('--use_fp16', dest = 'use_fp16',
                                action = 'store_true')
    feature_parser.add_argument('--use_fp32', dest = 'use_fp16',
                                action = 'store_false')
    parser.set_defaults(use_fp16 = False)

    parser.add_argument('--keep_prob', type = float, default = 0.5,
                        help = 'Keep probability for dropout')
    parser.add_argument('--num_hidden', type = int, default = 1024,
                        help = 'Number of hidden nodes')
    parser.add_argument('--num_rnn_layers', type = int, default = 2,
                        help = 'Number of recurrent layers')
    parser.add_argument('--checkpoint', type = str, default = None,
                        help = 'Continue training from checkpoint file')
    parser.add_argument('--rnn_type', type = str, default = 'didirectional',
                        help = 'unidirectional or bidirectional')
    parser.add_argument('--initial_lr', type = float, default = 0.00001,
                        help = 'Initial learning rate for training')
    parser.add_argument('--num_filters', type = int, default = 32,
                        help = 'Number of convolutional filters')
    parser.add_argument('--moving_avg_decay', type = float, default = 0.9999,
                        help = 'Decay to use for the moving average of weights')
    parser.add_argument('--num_epochs_per_decay', type = int, default = 5,
                        help = 'Epochs after which learning rate decays')
    parser.add_argument('--lr_decay_factor', type = float, default = 0.9,
                        help = 'Learning rate decay factor')
    parser.add_argument('--intra_op', type = int, default = 44,
                        help = 'Intra op thread num')
    parser.add_argument('--inter_op', type = int, default = 1,
                        help = 'Inter op thread num')
    parser.add_argument('--engine', type = str, default = 'rnn',
                        help = 'Select the engine you use')
 
    args = parser.parse_args()

    # Read architecture hyper-parameters from checkpoint file
    # if one is provided.
    if args.checkpoint is not None:
        param_file = os.path.join(args.checkpoint, 'deepSpeech_parameters.json')
        with open(param_file, 'r') as file:
            params = json.load(file)
            # Read network architecture parameters from previously saved
            # parameter file.
            args.num_hidden = params['num_hidden']
            args.num_rnn_layers = params['num_rnn_layers']
            args.rnn_type = params['rnn_type']
            args.num_filters = params['num_filters']
            args.use_fp16 = params['use_fp16']
            args.temporal_stride = params['temporal_stride']
            args.initial_lr = params['initial_lr']
            args.num_gpus = params['num_gpus']
            args.engine = params['engine']
    return args


def tower_loss(sess, scope, feats, labels, seq_lens):
    """Calculate the total loss on a single tower running the deepSpeech model.

    This function builds the graph for computing the loss per tower(GPU).

    ARGS:
      scope: unique prefix string identifying the
             deepSpeech tower, e.g. 'tower_0'
      feats: Tensor of shape BxFxT representing the
             audio features (mfccs or spectrogram).
      labels: sparse tensor holding labels of each utterance.
      seq_lens: tensor of shape [batch_size] holding
              the sequence length per input utterance.
    Returns:
       Tensor of shape [batch_size] containing
       the total loss for a batch of data
    """

    # Build inference Graph.
    logits = deepSpeech.inference(sess, feats, seq_lens, ARGS)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = deepSpeech.loss(logits, labels, seq_lens)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name = 'total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name = 'avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for loss in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a
        # multi-GPU training session. This helps the clarity
        # of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % helper_routines.TOWER_NAME, '',
                           loss.op.name)
        # Name each loss as '(raw)' and name the moving average
        # version of the loss as the original loss name.
        tf.summary.scalar(loss_name + '(raw)', loss)
        tf.summary.scalar(loss_name, loss_averages.average(loss))

    # Without this loss_averages_op would never run
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the
       gradient has been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for each_grad, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(each_grad, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # The variables are redundant because they are shared
        # across towers. So we will just return the first tower's pointer to
        # the Variable.
        weights = grad_and_vars[0][1]
        grad_and_var = (grad, weights)
        average_grads.append(grad_and_var)
    return average_grads


def set_learning_rate():
    """ Set up learning rate schedule """

    # Create a variable to count the number of train() calls.
    # This equals the number of batches processed * ARGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer = tf.constant_initializer(0), trainable = False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (deepSpeech.NUM_PER_EPOCH_FOR_TRAIN /
                             ARGS.batch_size)
    decay_steps = int(num_batches_per_epoch * ARGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
        ARGS.initial_lr,
        global_step,
        decay_steps,
        ARGS.lr_decay_factor,
        staircase = True)

    return learning_rate, global_step


def fetch_data():
    """ Fetch features, labels and sequence_lengths from a common queue."""
    # print "gpu num: ", ARGS.num_gpus

    tot_batch_size = ARGS.batch_size * ARGS.num_gpus
    feats, labels, seq_lens = deepSpeech.inputs(eval_data = 'train',
                                                data_dir = ARGS.data_dir,
                                                batch_size = tot_batch_size,
                                                use_fp16 = ARGS.use_fp16,
                                                shuffle = ARGS.shuffle)

    # Split features and labels and sequence lengths for each tower
    split_feats = tf.split(feats, ARGS.num_gpus, 0)
    split_labels = tf.sparse_split(sp_input = labels, num_split = ARGS.num_gpus, axis = 0)
    split_seq_lens = tf.split(seq_lens, ARGS.num_gpus, 0)

    return split_feats, split_labels, split_seq_lens


def get_loss_grads(sess, data, optimizer):
    """ Set up loss and gradient ops.
    Add summaries to trainable variables """

    # Calculate the gradients for each model tower.
    [feats, labels, seq_lens] = data
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for i in range(ARGS.num_gpus):
            with tf.device('/cpu:%d' % i):
                name_scope = '%s_%d' % (helper_routines.TOWER_NAME, i)
                with tf.name_scope(name_scope) as scope:
                    # Calculate the loss for one tower of the deepSpeech model.
                    # This function constructs the entire deepSpeech model
                    # but shares the variables across all towers.
                    loss = tower_loss(sess, scope, feats[i], labels[i], seq_lens[i])

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of
                    # data on this tower.
                    grads_and_vars = optimizer.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads_and_vars)

    return loss, tower_grads, summaries


def run_train_loop(sess, operations, saver):
    """ Train the model for required number of steps."""
    (train_op, loss_op, summary_op) = operations

    run_options = None
    run_metadata = None
    trace_file = None
    if DEBUG:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        trace_file = open('profiling.json', 'w')

    # Evaluate the ops for max_steps
    for step in range(ARGS.max_steps):
        start_time = time.time()
        
        if DUMMY:
            feats, idx, vals, shape, seq_lens = deepSpeech_dummy.inputs(ARGS.batch_size)
            data_gen_time = time.time()
            # labels_val = labels.eval(session = sess)
            dummy_input_duration = data_gen_time - start_time
            _, loss_value = sess.run([train_op, loss_op], options=run_options, run_metadata=run_metadata,
                                     feed_dict={feats_batch: feats,
                                                  idx_batch: idx, 
                                                  vals_batch: vals,
                                                  shape_batch: shape,
                                                  seq_lens_batch: seq_lens
                                                })
        else:
            _, loss_value = sess.run([train_op, loss_op], options=run_options, run_metadata=run_metadata)
 

        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step >= 10:
          profiling.append(duration)

        # Print progress periodically
        if step > 10 and step % 10 == 0:
            examples_per_sec = (ARGS.batch_size * ARGS.num_gpus) / np.average(profiling)
            if DUMMY:
                format_str = ('%s: step %d, '
                              'loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch; '
                              '%.3f dummy sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, np.average(profiling) / ARGS.num_gpus,
                                    dummy_input_duration))
            else:
                format_str = ('%s: step %d, '
                          'loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, np.average(profiling) / ARGS.num_gpus))

        # Run the summary ops periodically
        if step % 50 == 0 and not DUMMY:
            summary_writer = tf.summary.FileWriter(ARGS.train_dir, sess.graph)
            summary_writer.add_summary(sess.run(summary_op), step)

        # Save the model checkpoint periodically
        if step % 1000 == 0 or (step + 1) == ARGS.max_steps:
            checkpoint_path = os.path.join(ARGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step = step)

        if DEBUG and step == 20:
            trace = timeline.Timeline(run_metadata.step_stats)
            trace_file.write(trace.generate_chrome_trace_format())

            prof_options = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
            prof_options['dump_to_file'] = "./params.log"
            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                                                                tfprof_options = prof_options)
            # sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

            prof_options = tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS
            prof_options['dump_to_file'] = "./flops.log"
            tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                                                  run_meta = run_metadata,
                                                                  tfprof_options = prof_options)

            prof_options = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY
            prof_options['dump_to_file'] = "./timing_memory.log"
            prof_options['start_name_regexes'] = "ctc_loss"
            tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                                                  tfprof_cmd = 'graph',
                                                                  run_meta = run_metadata,
                                                                  tfprof_options = prof_options)


def initialize_from_checkpoint(sess, saver):
    """ Initialize variables on the graph"""
    # Initialise variables from a checkpoint file, if provided.
    ckpt = tf.train.get_checkpoint_state(ARGS.checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/train/model.ckpt-0,
        # extract global_step from it.
        checkpoint_path = ckpt.model_checkpoint_path
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        return global_step
    else:
        print('No checkpoint file found')
        return


def add_summaries(summaries, learning_rate, grads):
    """ Add summary ops"""

    # Track quantities for Tensorboard display
    summaries.append(tf.summary.scalar('learning_rate', learning_rate))
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)
    return summary_op


def train():
    """Train deepSpeech for a number of steps.
    This function build a set of ops required to build the model and optimize
    weights.

    """
    with g.as_default(), tf.device('/cpu'):
        # Learning rate set up
        learning_rate, global_step = set_learning_rate()

        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Fetch a batch worth of data for each tower
        if not DUMMY: 
            data = fetch_data()
        else: 
            # idx, vals, s_shape = deepSpeech_dummy.dense_to_sparse(labels_batch, labels_batch_w, labels_batch_h)
            labels_batch = tf.SparseTensor(indices = idx_batch, values = vals_batch, dense_shape = shape_batch)
            split_feats = tf.split(feats_batch, ARGS.num_gpus, 0)
            split_labels = tf.sparse_split(sp_input = tf.cast(labels_batch, tf.int32), num_split = ARGS.num_gpus, axis = 0)
            split_seq_lens = tf.split(seq_lens_batch, ARGS.num_gpus, 0)
            data = [split_feats, split_labels, split_seq_lens]

        # Start running operations on the Graph. allow_soft_placement
        # must be set to True to build towers on GPU, as some of the
        # ops do not have GPU implementations.
        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = ARGS.log_device_placement,
            inter_op_parallelism_threads = ARGS.inter_op,
            intra_op_parallelism_threads = ARGS.intra_op))

        # Construct loss and gradient ops
        loss_op, tower_grads, summaries = get_loss_grads(sess, data, optimizer)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # grads = tower_grads[0]

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads,
                                                      global_step = global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            ARGS.moving_avg_decay, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Build summary op
        summary_op = add_summaries(summaries, learning_rate, grads)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Initialize vars.
        if ARGS.checkpoint is not None:
            print "has checkpoint"
            global_step = initialize_from_checkpoint(sess, saver)
        else:
            print "does not have checkpoint"
            sess.run(tf.global_variables_initializer())

        if not DUMMY:
            # Start the queue runners.
            tf.train.start_queue_runners(sess)

        g.finalize()

        # Run training loop
        run_train_loop(sess, (train_op, loss_op, summary_op), saver)


def main():
    """
    Creates checkpoint directory to save training progress and records
    training parameters in a json file before initiating the training session.
    """
    if ARGS.train_dir != ARGS.checkpoint:
        if tf.gfile.Exists(ARGS.train_dir):
            tf.gfile.DeleteRecursively(ARGS.train_dir)
        tf.gfile.MakeDirs(ARGS.train_dir)

    # Dump command line arguments to a parameter file,
    # in-case the network training resumes at a later time.
    with open(os.path.join(ARGS.train_dir,
                           'deepSpeech_parameters.json'), 'w') as outfile:
        json.dump(vars(ARGS), outfile, sort_keys=True, indent=4)

    args = setenvs(sys.argv)
    print('Running on platform: ', args.platform)
  
    if args.platform == 'bdw':
        ARGS.intra_op = 8
        ARGS.inter_op = 5
    elif args.platform == 'knl':
        ARGS.intra_op = 8
        ARGS.inter_op = 8

    print('Running inter_op: ', ARGS.inter_op)
    print('Running intra_op: ', ARGS.intra_op)
 
    train()

if __name__ == '__main__':
    ARGS = parse_args()
    main()
