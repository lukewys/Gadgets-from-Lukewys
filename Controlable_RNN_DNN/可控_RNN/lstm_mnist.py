import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

start_time = time.time()


def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


# 设置 GPU 按需增长，如果用cpu就删除这一段
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

learn = tf.contrib.learn
HIDDEN_SIZE = 128  # Lstm中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数
TIMESTEPS = 28  # 循环神经网络的截断长度
n_input = 28
TRAINING_STEPS = 10000  # 训练轮数
BATCH_SIZE = 100  # batch大小
N_ONEHOT = 10


def generate_data(seq):
    X = []
    Y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;第i+TIMESTEPS项作为输出
    # 即用sin函数前面的TIMESTPES个点的信息，预测第i+TIMESTEPS个点的函数值
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    return lstm_cell


weights = {
    'out': tf.Variable(tf.random_normal([HIDDEN_SIZE, N_ONEHOT])),
    'in': tf.Variable(tf.random_normal([n_input, HIDDEN_SIZE]))
}
biases = {
    'out': tf.Variable(tf.random_normal([N_ONEHOT])),
    'in': tf.Variable(tf.random_normal([HIDDEN_SIZE]))
}

x = tf.placeholder("float", [None, TIMESTEPS, 28])
y = tf.placeholder("float", [None, N_ONEHOT])


def RNN(x, weights, biases):
    # 规整输入的数据
    x = tf.transpose(x, [1, 0, 2])  # permute n_steps and batch_size
    print(x.shape)
    x = tf.reshape(x, [-1, n_input])  # (n_steps*batch_size, n_input)
    print(x.shape)
    # 输入层到隐含层，第一次是直接运算
    x = tf.matmul(x, weights['in']) + biases['in']
    x = tf.split(x, 28, 0)
    print(x)

    rnn_cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    a1 = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return a1


y_ = RNN(x, weights, biases)
y = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

testimage = mnist.test.images.reshape(10000, 28, 28)
#
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs = batch_xs.reshape((100, 28, n_input))
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    ACC = np.array([], dtype=np.float32)
    if i % 200 == 0:
        print("i=" + str(i))
        AC = sess.run(accuracy, feed_dict={x: testimage, y: mnist.test.labels})
        print("acc=", AC)
        ACC = np.append(ACC, AC)
print("Elapsed time: ", elapsed(time.time() - start_time))
