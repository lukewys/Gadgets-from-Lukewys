import numpy as np
from tensorflow.contrib import rnn
from matplotlib import pyplot as plt
import datetime
import random
import pretty_midi
import ast
import os
import time
import tensorflow as tf
import magenta



global batch_count
batch_count = 0
global epoch_count
epoch_count = 0
RANDOM_OUT = True
HIDDEN_SIZE = 128
NUM_LAYERS = 2
TIMESTEPS = 256
learning_rate = 0.00001
ONEHOT_SIZE = 128
RECUR_STEP = 256
sess = tf.InteractiveSession()


def clear_count():
    global batch_count
    batch_count = 0
    global epoch_count
    epoch_count = 0


def elapsed(sec):
    if sec < 60:
        return "{:.2f}".format(sec) + " sec"
    elif sec < (60 * 60):
        return "{:.2f}".format(sec / 60) + " min"
    else:
        return "{:.2f}".format(sec / (60 * 60)) + " hr"

#start_time=time.time()
#print("Elapsed time: ", elapsed(time.time() - start_time))

def make_layer(INPUT_SIZE,OUTPUT_SIZE,HIDDEN_SIZE,LAYER_NUMBER,BOOL):
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    weights={'W0':tf.Variable(tf.truncated_normal([INPUT_SIZE, HIDDEN_SIZE], stddev=0.1),trainable=BOOL)}
    bias={'b0':tf.Variable(tf.constant(0.1, shape=[HIDDEN_SIZE]),trainable=BOOL)}
    a={'a0':tf.nn.tanh(tf.matmul(x, weights['W0']) + bias['b0'])}
    for i in range(1,LAYER_NUMBER+1):
        weights['W'+str(i)] = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, HIDDEN_SIZE], stddev=0.1),trainable=BOOL)
        bias['b' + str(i)] = tf.Variable(tf.constant(0.1, shape=[HIDDEN_SIZE]),trainable=BOOL)
        a['a' + str(i)] = tf.nn.tanh(tf.matmul(a['a' + str(i-1)],weights['W'+str(i)]) + bias['b' + str(i)])

    weights['W' + str(LAYER_NUMBER+1)] = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, OUTPUT_SIZE], stddev=0.1),trainable=BOOL)
    bias['b' + str(LAYER_NUMBER+1)] = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]),trainable=BOOL)
    y = tf.nn.sigmoid(tf.matmul(a['a' + str(LAYER_NUMBER)],weights['W'+str(LAYER_NUMBER+1)])
                      + bias['b' + str(LAYER_NUMBER+1)])
    return x,y



def generate_data(seq, TIMESTEPS):
    X = []
    Y = []
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def get_data(num, data, TIMESTEPS):
    if num == -1:
        batch_x, batch_y = generate_data(data, TIMESTEPS)
        batch_x = batch_x.reshape(-1, TIMESTEPS, 1)
        return batch_x, batch_y
    n = len(data)
    r = np.random.random_integers(0, n - num - TIMESTEPS - 2)
    seq = data[r:r + num + TIMESTEPS + 1]
    batch_x, batch_y = generate_data(seq, TIMESTEPS)
    batch_x = batch_x.reshape(-1, TIMESTEPS, 1)
    return batch_x, batch_y


def get_data_n(num, data, TIMESTEPS):
    if num == -1:
        batch_x, batch_y = generate_data(data, TIMESTEPS)
        batch_x = batch_x.reshape(-1, TIMESTEPS, 1)
        return batch_x, batch_y
    global batch_count
    global epoch_count
    n = len(data)
    r = batch_count % (n - num - TIMESTEPS - 2)
    seq = data[r:r + num + TIMESTEPS + 1]
    batch_x, batch_y = generate_data(seq, TIMESTEPS)
    batch_x = batch_x.reshape(-1, TIMESTEPS, 1)
    FIRST = batch_count == 0 and epoch_count == 0
    batch_count += num
    if r < num and not FIRST:
        epoch_count += 1
        print('EPOCH ' + str(epoch_count))
    return batch_x, batch_y


def make_batch(num, data, ONEHOT_SIZE, TIMESTEPS):
    batch_xs, batch_ys = get_data(num, data, TIMESTEPS)
    batch_ys = onehot_gen(batch_ys, ONEHOT_SIZE).reshape((-1, ONEHOT_SIZE))
    return batch_xs, batch_ys


def make_batch_onehot(num, data, ONEHOT_SIZE, TIMESTEPS):
    batch_xs, batch_ys = get_data(num, data, TIMESTEPS)
    batch_xs = onehot_gen(batch_xs, ONEHOT_SIZE).reshape((-1, TIMESTEPS, ONEHOT_SIZE))
    batch_ys = onehot_gen(batch_ys, ONEHOT_SIZE).reshape((-1, ONEHOT_SIZE))
    return batch_xs, batch_ys


def make_batch_n(num, data, ONEHOT_SIZE, TIMESTEPS):
    batch_xs, batch_ys = get_data_n(num, data, TIMESTEPS)
    batch_ys = onehot_gen(batch_ys, ONEHOT_SIZE).reshape((-1, ONEHOT_SIZE))
    return batch_xs, batch_ys


def make_batch_n_onehot(num, data, ONEHOT_SIZE, TIMESTEPS):
    batch_xs, batch_ys = get_data_n(num, data, TIMESTEPS)
    batch_xs = onehot_gen(batch_xs, ONEHOT_SIZE).reshape((-1, TIMESTEPS, ONEHOT_SIZE))
    batch_ys = onehot_gen(batch_ys, ONEHOT_SIZE).reshape((-1, ONEHOT_SIZE))
    return batch_xs, batch_ys


def onehot_gen(data, ONEHOT_SIZE):
    data = data.reshape(-1)
    m = ONEHOT_SIZE
    n = data.shape[0]
    p = np.zeros([n, m], dtype=np.float32)
    for i in range(n):
        l = int(data[i])
        p[i, l] = 1
    return p


def random_index(rate):
    start = 0
    randnum = random.uniform(0, np.sum(rate))
    for index in range(len(rate)):
        start += rate[index]
        if randnum <= start:
            break
    return index


def LstmCell(HIDDEN_SIZE):
    # lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE)
    lstm_cell = rnn.GRUCell(HIDDEN_SIZE)
    return lstm_cell


def RNN(x, weights, biases, HIDDEN_SIZE, TIMESTEPS, NUM_LAYERS):
    x = tf.reshape(x, [-1, TIMESTEPS])
    x = tf.split(x, TIMESTEPS, 1)
    rnn_cell = rnn.MultiRNNCell([LstmCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def RNN_onehot_in(x, weights, biases, HIDDEN_SIZE, ONEHOT_SIZE, TIMESTEPS, NUM_LAYERS):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, ONEHOT_SIZE])
    x = tf.matmul(x, weights['in']) + biases['in']
    x = tf.split(x, TIMESTEPS, 0)
    rnn_cell = rnn.MultiRNNCell([LstmCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def save_acc_loss(ACC, LOSS):
    now_time = datetime.datetime.now()
    str_time = datetime.datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    np.save("results/loss_" + str_time + ".npy", LOSS)
    np.save("results/acc_" + str_time + ".npy", ACC)
    plt.figure('loss')
    plt.plot(LOSS)
    plt.savefig("results/loss_" + str_time + ".png")
    plt.figure('accuracy')
    plt.plot(ACC)
    plt.savefig("results/accuracy_" + str_time + ".png")


class note:

    def __init__(self, key, duration):
        self.key = key
        self.duration = duration

    def show(self):
        print('key:' + str(self.key))
        print("duration:" + str(self.duration))


def data_to_note(data):
    notelist = []
    n = len(data)
    offset = 0
    data[-1] = 0
    while offset < n - 1:
        if data[offset] != 0:
            num = 0
            key = data[offset]
            while data[offset] == key:
                num += 1
                offset += 1
                if data[offset] == key and data[offset + 1] != key:
                    p = note(key, num)
                    notelist.append(p)
        else:
            offset += 1
    return notelist


def to_midi(velocity, rate, name, data):
    data = data.astype(int)
    l = data_to_note(data)
    offset = 0
    midi = pretty_midi.PrettyMIDI()
    midi_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    midi_instrument = pretty_midi.Instrument(program=midi_program)
    for _ in range(len(l)):
        note = (pretty_midi.Note(velocity, l[_].key, offset, offset + l[_].duration * rate))
        offset += l[_].duration * rate
        midi_instrument.notes.append(note)
    midi.instruments.append(midi_instrument)
    midi.write(name)


x = tf.placeholder("float", [None, TIMESTEPS, ONEHOT_SIZE])
y = tf.placeholder("float", [None, ONEHOT_SIZE])

weights = {
    'out': tf.Variable(tf.random_normal([HIDDEN_SIZE, ONEHOT_SIZE])),
    'in': tf.Variable(tf.random_normal([ONEHOT_SIZE, HIDDEN_SIZE]))
}
biases = {
    'out': tf.Variable(tf.random_normal([ONEHOT_SIZE])),
    'in': tf.Variable(tf.random_normal([HIDDEN_SIZE]))
}

pred = RNN_onehot_in(x, weights, biases, HIDDEN_SIZE, ONEHOT_SIZE, TIMESTEPS, NUM_LAYERS)
softmax_pred = tf.nn.softmax(pred)
# softmax_pred = tf.exp(pred) / tf.reduce_sum(tf.exp(pred))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)


def recur(N, data):
    if len(data) > TIMESTEPS:
        data = data[:TIMESTEPS]
    if len(data) < TIMESTEPS:
        l = len(data)
        z = np.zeros((TIMESTEPS - l))
        data = np.append(data, z)
    predict = np.array([], dtype=float)
    predict = np.append(predict, data.reshape(-1))
    recur_time = time.time()
    model_file = tf.train.latest_checkpoint('save/')
    saver.restore(sess, model_file)
    i = 0
    while i < N:
        xs = onehot_gen(data, ONEHOT_SIZE).reshape((-1, TIMESTEPS, ONEHOT_SIZE))
        if RANDOM_OUT == True:
            pred_ = sess.run(softmax_pred, feed_dict={x: xs}).reshape(-1)
            pred_ = random_index(pred_)
        else:
            pred_ = sess.run(pred, feed_dict={x: xs})
            pred_ = pred_.argmax(axis=1)
        predict = np.append(predict, pred_)
        data = np.append(data, pred_)
        data = data[1:]
        i += 1
    return predict


def generate_mono(filepath, times):
    FS = 32
    p = np.array([], dtype=int)
    midi_data = pretty_midi.PrettyMIDI(filepath)
    a = midi_data.instruments[0].get_piano_roll(fs=FS)
    datrc = np.append(p, np.argmax(a, axis=0))  # 取高音
    velocity = 80
    rate = 0.015625 * 2
    print("\n\n\n开始生成...\n\n\n")
    for _ in range(times):
        predicted = recur(RECUR_STEP, datrc)
        now_time = datetime.datetime.now()
        str_time = datetime.datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
        abspath = os.path.abspath('.')
        outdir = abspath + '\\results\\mono\\'
        to_midi(velocity, rate, outdir+str_time+'.mid', predicted)

'''
x = tf.placeholder(tf.float32, [None, 784])
w1 = tf.Variable(tf.truncated_normal([784, 5000], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[5000]))
a1 = tf.nn.relu(tf.matmul(x, w1) + b1)
w2 = tf.Variable(tf.truncated_normal([5000, 2000], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[2000]))
a2 = tf.nn.relu(tf.matmul(a1, w2) + b2)
w3 = tf.Variable(tf.truncated_normal([2000, 10], stddev=0.1))

b = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.nn.softmax(tf.matmul(a2, w3) + b)
'''