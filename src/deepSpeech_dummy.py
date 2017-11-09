import random
import numpy as np
import tensorflow as tf


# counts = [3, 10, 11, 13, 14, 13, 9, 8, 5, 4, 3, 2, 2, 2, 1]
# label_lengths = [7, 17, 35, 48, 62, 78, 93, 107, 120, 134, 148, 163, 178, 193, 209]

utt_lengths = [100, 200, 100, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
counts = [30, 3, 3, 3, 14, 13, 9, 8, 5, 4, 3, 2, 2, 2, 1]
label_lengths = [7, 17, 7, 48, 62, 78, 93, 107, 120, 134, 148, 163, 178, 193, 209]

freq_bins = 161
# scale_factor = 10 * 128
scale_factor = 10
extra = 1000

g_utter_counts = [x * scale_factor for x in counts]
g_batch_size = 0
g_randomness = np.zeros((1, freq_bins))
g_size = 0
g_duration = 0
g_current = 0


NUM_CLASSES = 29
NUM_PER_EPOCH_FOR_TRAIN = sum(counts) * scale_factor
# NUM_PER_EPOCH_FOR_TRAIN = 28535
NUM_PER_EPOCH_FOR_EVAL = 2703
NUM_PER_EPOCH_FOR_TEST = 2620

def _init_data(batch_size):
    global g_batch_size
    global g_randomness
    global g_utter_counts
    global g_size
    global g_duration
    if g_batch_size != batch_size:
        # print 'set new batch_size %d' % (batch_size)
        g_current = 0
        # g_utter_counts = [x * scale_factor for x in counts]
        g_utter_counts = [x * scale_factor * batch_size for x in counts]
        g_batch_size = batch_size
        line = batch_size * (utt_lengths[-1] + extra)
        # print g_randomness.shape
        np.resize(g_randomness, (line, freq_bins))
        g_randomness = np.random.randn(line, freq_bins)
        g_size = 0
        g_duration = 0
        for idx, val in enumerate(g_utter_counts):
            g_size = g_size + val
            g_duration = g_duration + val * utt_lengths[idx] / 100

def _next(batch_size):
    _init_data(batch_size)
    global g_utter_counts
    global g_current
    if g_current >= len(g_utter_counts):
        print "One Epoch Complete"
        g_current = 0
        g_utter_counts = [x * scale_factor * batch_size for x in counts]

    inc = 0
    l_batch_size = 0

    if (g_utter_counts[g_current] > batch_size):
        l_batch_size = batch_size
        g_utter_counts[g_current] = g_utter_counts[g_current] - batch_size
        inc = 0
    else:
        l_batch_size = g_utter_counts[g_current]
        g_utter_counts[g_current] = 0
        inc = 1
    # print 'utter counts %d' % g_utter_counts[g_current]
    utt_length = utt_lengths[g_current]
    label_length = label_lengths[g_current]
    start_idx = random.randint(0, extra + batch_size * (utt_lengths[-1] - utt_lengths[g_current]) - 1)
    end_idx = start_idx + utt_length * l_batch_size

    g_current = g_current + inc
    label = range(label_length)
    for x in range(label_length - 1):
        label[x] = random.randint(0, NUM_CLASSES - 2)
    label[label_length - 1] = NUM_CLASSES - 1
    feat = g_randomness[start_idx : end_idx, :]

    return utt_length, feat, label

def _dense_to_sparse(dense):
    idx = []
    val = []
    for l in range(dense.shape[0]):
        for c in range(dense.shape[1]):
            if dense[l, c] != 0:
                val.append(dense[l, c])
                idx.append([l, c])
    # print idx, val, dense.shape
    return idx, val, dense.shape


def inputs(batch_size):
    """Construct input for dummy data

    Returns:
      feats: MFCC. 3D tensor of [batch_size, T, F] size.
      labels: Labels. 1D tensor of [batch_size] size.
      seq_lens: SeqLens. 1D tensor of [batch_size] size.
    """

    utt_length, feat, label = _next(batch_size)
    seq_lens = np.full(batch_size, utt_length)

    feats = np.reshape(feat, [batch_size, utt_length, freq_bins])

    labels = np.zeros((batch_size, len(label)))
    for x in range(batch_size):
        labels[x] = label
    idx, vals, s_shape = _dense_to_sparse(labels)
    # t_labels = tf.SparseTensor(indices = idx, values = vals, dense_shape = s_shape)
    # print feats, t_labels, seq_len
    # return feats.astype(np.float32), tf.cast(t_labels, tf.int32), seq_lens.astype(np.int32)
    return feats.astype(np.float32), idx, vals, s_shape, seq_lens.astype(np.int32)

