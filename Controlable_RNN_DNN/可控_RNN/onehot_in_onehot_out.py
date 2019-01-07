import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import datetime
import sys

sys.path.append(r'E:\机器学习加音乐\主程序\单声部_单音预测\Lib_AIMusic')
import AIMusic_package as pk
from AIMusic_package import *


# TODO: tomidi改成可存文件路径，验证random_index正确性 两种softmax的不同 放弃全0的训练输入


def recur(N, data):
    print("Start Recur")
    print(len(data))
    if len(data) != TIMESTEPS:
        print("DATA LENGTH ERROR!")
        return 0
    predict = np.array([], dtype=float)
    predict = np.append(predict, data.reshape(-1))
    recur_time = time.time()
    i = 0
    while i < N:
        # for i in range(N):
        xs = onehot_gen(data, ONEHOT_SIZE).reshape((-1, TIMESTEPS, ONEHOT_SIZE))
        if RANDOM_OUT == True:
            #pred_ = sess.run(softmax_pred, feed_dict={x: xs}).reshape(-1)
            pred_ = sess.run(tf.nn.softmax(pred), feed_dict={x: xs}).reshape(-1)
            pred_ = random_index(pred_)
        else:
            pred_ = sess.run(pred, feed_dict={x: xs})
            pred_ = pred_.argmax(axis=1)
        predict = np.append(predict, pred_)
        data = np.append(data, pred_)
        data = data[1:]
        if len(predict) == 50 + TIMESTEPS:
            print("Recur time_50: ", elapsed(time.time() - recur_time))
        i += 1
    return predict


HIDDEN_SIZE = 128
NUM_LAYERS = 2
TIMESTEPS = 256
TRAINING_STEPS = 5000000
# data_single_piano epoch20: STEP=867000
BATCH_SIZE = 400
learning_rate = 0.00001
display_step = 1000
ONEHOT_SIZE = 128
TRAIN = False
CONTINUE = True
SAVE_LOSS_ACC = True
RANDOM_OUT = True
out_mode = 'recur'
RECUR_WITH_DATA_SET = False
# out_mode = 'out_test_set'
RECUR_STEP = 256
velocity = 80
rate = 0.015625 * 2
name = 'b.mid'

if TRAIN==True:
    datr = np.load(r"E:\机器学习加音乐\主程序\单声部_单音预测\dataset\bach_13W_fs32.npy")
    #datr = np.load(r"E:\机器学习加音乐\主程序\单声部_单音预测\dataset\data_single_piano.npy")
    datasize = len(datr)
    print("Dataset size:" + str(datasize))
    datv = np.load(r"E:\机器学习加音乐\主程序\单声部_单音预测\dataset\bach_13W_fs32.npy")
    # datv = datr
    dats = datv[1000:50000]

    # datrc_start = 1000
    # datrc = dats[datrc_start:TIMESTEPS + datrc_start]

    print("Data loaded")

datrc = np.load("star.npy")
sess = tf.InteractiveSession()
start_time = time.time()

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
if TRAIN == True:
    sess.run(init)
    acc_total = 0
    loss_total = 0
    ACC = np.array([], dtype=np.float32)
    LOSS = np.array([], dtype=np.float32)
    if CONTINUE == True:
        model_file = tf.train.latest_checkpoint('save_onehot_in/')
        i = model_file.find('-')
        num = model_file[i + 1:]
        print(num)
        saver.restore(sess, model_file)
        print("Model loaded:" + str(model_file))
        print('data_size=' + str(datasize))
        i = int(num)
        pk.batch_count = (i * BATCH_SIZE) % datasize
        pk.epoch_count = int((i * BATCH_SIZE) / datasize)
        print('batch_count = ' + str(pk.batch_count))
        print('epoch_count = ' + str(pk.epoch_count))
    else:
        i = 0
    saver_path = saver.save(sess, "save_onehot_in/model.ckpt", global_step=i)
    print("Training...")
    time_now = time.time()
    for i in range(i, TRAINING_STEPS):
        batch_xs, batch_ys = make_batch_n_onehot(BATCH_SIZE, datr, ONEHOT_SIZE, TIMESTEPS)
        _, acc, loss = sess.run([optimizer, accuracy, cost], feed_dict={x: batch_xs, y: batch_ys})
        loss_total += loss
        acc_total += acc
        if i == 0:
            acc_zero = np.mean(batch_ys[:, 0])
            print("loss=" + "{:.6f}".format(loss) +
                  ", Iter= " + str(i) +
                  ", Average Loss= " + "{:.6f}".format(loss_total / display_step) +
                  ", Average Accuracy= " +
                  "{:.2f}%".format(100 * acc_total / display_step) +
                  ", Zero Predict Accuracy:" +
                  "{:.2f}%".format(100 * acc_zero) +
                  ", time=" + elapsed(time.time() - time_now))
            time_now = time.time()
        if i % 50 == 0:
            ACC = np.append(ACC, acc)
            LOSS = np.append(LOSS, loss)
        if (i + 1) % display_step == 0:
            acc_zero = np.mean(batch_ys[:, 0])
            print("loss=" + "{:.6f}".format(loss) +
                  ", Iter= " + str(i + 1) +
                  ", Average Loss= " + "{:.6f}".format(loss_total / display_step) +
                  ", Average Accuracy= " +
                  "{:.2f}%".format(100 * acc_total / display_step) +
                  ", Zero Predict Accuracy:" +
                  "{:.2f}%".format(100 * acc_zero) +
                  ", time=" + elapsed(time.time() - time_now))
            time_now = time.time()
            acc_total = 0
            loss_total = 0
        if (i - 1) % (display_step * 3) == 0 and (i - 1) != 0:
            batch_xs, batch_ys = make_batch_onehot(BATCH_SIZE * 5, dats, ONEHOT_SIZE, TIMESTEPS)
            acc_v, loss_v = sess.run([accuracy, cost], feed_dict={x: batch_xs, y: batch_ys})
            print("loss_V=" + "{:.6f}".format(loss_v) + ", Iter= " + str(i - 1) + ", Accuracy= " +
                  "{:.2f}%".format(100 * acc_v))
            saver.save(sess, 'save_onehot_in/model.ckpt', global_step=(i - 1))
    if SAVE_LOSS_ACC == True:
        save_acc_loss(ACC, LOSS)

    print("Training complete!")
    print("Elapsed time: ", elapsed(time.time() - start_time))

else:
    model_file = tf.train.latest_checkpoint('save_onehot_in/')
    saver.restore(sess, model_file)
    print("Model loaded:" + str(model_file))

if out_mode == 'out_test_set':
    clear_count()
    predicted = np.array([])
    test_y = np.array([])
    while pk.epoch_count < 1:
        batch_xs, batch_ys = make_batch_n_onehot(100, dats, ONEHOT_SIZE, TIMESTEPS)
        batch_ys = batch_ys.argmax(axis=1)
        test_y = np.append(test_y, batch_ys.reshape(-1))
        pred_ = sess.run(pred, feed_dict={x: batch_xs})
        predicted = np.append(predicted, pred_.argmax(axis=1))
    plt.figure('result')
    plot_test, = plt.plot(test_y, 'b.', label='data_sample')
    plot_predicted, = plt.plot(predicted, 'g.', label='predicted')
    plt.legend([plot_predicted, plot_test], ['predicted', 'test_set'])
    now_time = datetime.datetime.now()
    str_time = datetime.datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    plt.savefig("results/result.png_" + str_time + ".png")
    plt.show()

if out_mode == 'recur':
    predicted = recur(RECUR_STEP, datrc)
    plt.figure('result_recur')
    if RECUR_WITH_DATA_SET == True:
        test_y = dats[datrc_start:datrc_start + TIMESTEPS + RECUR_STEP]
        plot_test, = plt.plot(test_y, 'b.', label='data_sample')
        plot_predicted, = plt.plot(predicted, 'g.', label='predicted')
        plt.legend([plot_predicted, plot_test], ['predicted', 'test_set'])
    else:
        plot_predicted, = plt.plot(predicted, 'g.', label='predicted')
        plt.legend([plot_predicted], ['predicted'])
    now_time = datetime.datetime.now()
    str_time = datetime.datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    plt.savefig("results/result_recur_" + str_time + ".png")
    np.save("results/result_recur_" + str_time + ".npy", predicted)
    to_midi(velocity, rate, name, predicted)
    plt.show()
