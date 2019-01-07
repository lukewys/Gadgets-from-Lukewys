import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import datetime
import sys
from tensorflow.contrib import rnn
import random

import pandas as pd



HIDDEN_SIZE = 512
NUM_LAYERS = 2
TRAINING_STEPS = 100000
learning_rate = 0.003

display_step = 1000
ONEHOT_SIZE = 33*24
TRAIN = False
CONTINUE = False
SAVE_LOSS_ACC = True
out_mode = 'out_test_set'
out_mode = ''
TIMESTEPS_IN = 56
TIMESTEPS_OUT = 31
BATCH_SIZE = 2
global batch_count
batch_count = 0
global epoch_count
epoch_count = 0

data = pd.read_csv("full.csv").as_matrix(columns=None)
data = data[:, 5].reshape(-1, ONEHOT_SIZE)
#data=np.loadtxt('sin_10k_rand_0.1.txt')[:,1]
#data=np.load('3456.npy')
print(len(data))
print(data.shape)



data=np.log1p(data)


#plt.plot((data),'.')

'''mean = np.mean(data)
std = np.std(data)
data = (data - mean) / std

print('mean=' + str(mean))
print('std=' + str(std))
'''






def elapsed(sec):
    if sec < 60:
        return "{:.2f}".format(sec) + " sec"
    elif sec < (60 * 60):
        return "{:.2f}".format(sec / 60) + " min"
    else:
        return "{:.2f}".format(sec / (60 * 60)) + " hr"


def generate_data_seq2seq(seq, TIMESTEPS_IN, TIMESTEPS_OUT):
    X = []
    Y = []
    for i in range(len(seq) - TIMESTEPS_IN - TIMESTEPS_OUT):
        X.append([seq[i:i + TIMESTEPS_IN]])
        Y.append([seq[i + TIMESTEPS_IN:i + TIMESTEPS_IN + TIMESTEPS_OUT]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
global r
r=0

def get_data_n_seq2seq_24(num, data_xs, data_ys):
    global batch_count
    global epoch_count
    global r
    batch_x=data_xs[r:r + 1]
    batch_y = data_ys[r:r + 1]
    r+=24
    r = r % data_xs.shape[0]
    for i in range(num-1):
        batch_x = np.append(batch_x,data_xs[r:r + 1],axis=0)
        batch_y = np.append(batch_y,data_ys[r:r + 1],axis=0)
        r += 24
    FIRST = batch_count == 0 and epoch_count == 0
    batch_count += num
    if r < num and not FIRST:
        epoch_count += 1
        print('EPOCH ' + str(epoch_count))
    return batch_x, batch_y

def get_data_n_seq2seq(num, data_xs, data_ys):
    global batch_count
    global epoch_count
    global r

    r = r % data_xs.shape[0]
    batch_x = data_xs[r:r + num]
    batch_y = data_ys[r:r + num]
    FIRST = batch_count == 0 and epoch_count == 0
    batch_count += num
    r += num
    if r < num and not FIRST:
        epoch_count += 1
        print('EPOCH ' + str(epoch_count))
    return batch_x, batch_y


xs, ys = generate_data_seq2seq(data, TIMESTEPS_IN, TIMESTEPS_OUT)
xs=xs.reshape(-1,TIMESTEPS_IN,ONEHOT_SIZE)
ys=ys.reshape(-1,TIMESTEPS_OUT,ONEHOT_SIZE)
print(xs.shape)
print(ys.shape)


'''
xt, yt = get_data_n_seq2seq(1, xs, ys)
xt=xt[:,:,0].reshape(-1)
yt=yt[:,:,0].reshape(-1)
plt.plot(xt,'.')
plt.show()
plt.plot(yt,'.')
plt.show()
'''


encoder_inputs = tf.placeholder(shape=(BATCH_SIZE, TIMESTEPS_IN, ONEHOT_SIZE), dtype=tf.float32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(BATCH_SIZE, TIMESTEPS_OUT, ONEHOT_SIZE), dtype=tf.float32,
                                 name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(BATCH_SIZE, 1, ONEHOT_SIZE), dtype=tf.float32, name='decoder_inputs')
decoder_seq_length = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='batch_seq_length')

encoder_cell = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs,
    dtype=tf.float32)
del encoder_outputs

start_tokens = np.array([0]).repeat(BATCH_SIZE)
end_token = -1

decoder_embedding = tf.Variable(tf.truncated_normal(shape=[ONEHOT_SIZE,ONEHOT_SIZE], stddev=0.1),
                                name='decoder_embedding')
from tensorflow.python.layers import core
import tensorflow.contrib.layers as layers

output_layer = core.Dense(
    ONEHOT_SIZE, activation=tf.nn.relu, use_bias=True, name="output_projection",
    kernel_initializer=layers.variance_scaling_initializer(factor=1.0,
                                                           uniform=True,
                                                           seed=1))


decoder_cell = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    decoder_embedding, start_tokens, end_token)

decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_final_state, output_layer=output_layer)

outputs, final_context_state = tf.contrib.seq2seq.dynamic_decode(
    decoder=decoder, maximum_iterations=TIMESTEPS_OUT, swap_memory=True)

pred=outputs.rnn_output
'''
pc=outputs.rnn_output
w=tf.Variable(tf.random_normal([ONEHOT_SIZE, ONEHOT_SIZE]))
b=tf.Variable(tf.random_normal([ONEHOT_SIZE]))
pc = tf.reshape(pc, [-1, ONEHOT_SIZE])
pred = tf.matmul(pc,w)+b
pred=tf.reshape(pred, [-1,TIMESTEPS_OUT, ONEHOT_SIZE])
'''

cost = tf.losses.mean_squared_error(predictions=pred, labels=decoder_targets)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.InteractiveSession()
start_time = time.time()
init = tf.global_variables_initializer()
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

isExists = os.path.exists('save_seq2seq_full')
if not isExists:
    os.makedirs('save_seq2seq_full')

save_dir = 'save_seq2seq_full/'

if TRAIN == True:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    sess.run(init)
    i = 0
    loss_total = 0
    saver_path = saver.save(sess, save_dir + 'model.ckpt', global_step=i)
    print("Training...")
    time_now = time.time()
    for i in range(TRAINING_STEPS):

        batch_xs, batch_ys = xs, ys
        #print(batch_xs.shape)
        #print(batch_ys.shape)
        _, loss = sess.run([optimizer, cost], feed_dict={encoder_inputs: batch_xs, decoder_targets: batch_ys})
        loss_total += loss
        if i == 0:
            print("loss=" + "{:.6f}".format(loss) +
                  ", Iter= " + str(i) +
                  ", Average Loss= " + "{:.6f}".format(loss_total / display_step) +
                  ", time=" + elapsed(time.time() - time_now))
            time_now = time.time()
            empty = np.array([])
            np.save(str(loss)+'.npy',empty)
        if i==10000:
            learning_rate = 0.00007
        if (i + 1) % display_step == 0:
            print("loss=" + "{:.6f}".format(loss) +
                  ", Iter= " + str(i + 1) +
                  ", Average Loss= " + "{:.6f}".format(loss_total / display_step) +
                  ", time=" + elapsed(time.time() - time_now))
            time_now = time.time()
            acc_total = 0
            loss_total = 0
            saver.save(sess, save_dir + 'model.ckpt', global_step=(i - 1))

    print("Training complete!")
    print("Elapsed time: ", elapsed(time.time() - start_time))

else:
    model_file = tf.train.latest_checkpoint('save_seq2seq_full')
    saver.restore(sess, model_file)
    print("Model loaded:" + str(model_file))

isExists = os.path.exists('save_seq2seq_full')
if not isExists:
    os.makedirs('save_seq2seq_full')


if out_mode == 'out_test_set':
    predicted = np.array([])
    batch_xs=data[-TIMESTEPS_IN-TIMESTEPS_OUT:-TIMESTEPS_OUT].reshape(1,TIMESTEPS_IN,ONEHOT_SIZE)
    batch_ys=data[-TIMESTEPS_OUT:].reshape(1,TIMESTEPS_OUT,ONEHOT_SIZE)
    test_y = batch_ys
    pred_ = sess.run(pred, feed_dict={encoder_inputs: batch_xs})
    np.save('results_seq2seq_3456/pred_seq2seq.npy', pred_)
    np.save('results_seq2seq_3456/y_seq2seq.npy', test_y)

data_t=np.append(xs[-1],xs[-1],axis=0).reshape(-1,TIMESTEPS_IN,ONEHOT_SIZE)
#data_t = np.append(data[-TIMESTEPS_IN-TIMESTEPS_IN:-TIMESTEPS_IN].reshape(1,TIMESTEPS_IN,ONEHOT_SIZE),
                   #data[-TIMESTEPS_IN-TIMESTEPS_IN:-TIMESTEPS_IN].reshape(1, TIMESTEPS_IN, ONEHOT_SIZE),axis=0)
print(data_t.shape)# [BATCH_SIZE,TIMESTEPS_IN,ONEHOT_SIZE]
pred_ = sess.run(tf.exp(pred)+1, feed_dict={encoder_inputs: data_t})
#pred_=pred_*(data.max()-data.min())+data.min()
#pred_ = np.exp(pred_)+1
np.save('finalresults1.npy', pred_)

