import os
import model
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("/home/shiva/junk/tf-learning/mnist/mnist/input_data", one_hot=True)

LEARNING_RATE = 1e-4
N_TRAIN_STEPS = 20000
BATCH_SIZE = 100
# model
x = tf.placeholder(tf.float32, [None, 784])

with tf.variable_scope("perceptron"):
    y_percep, perceptron_variables = model.multilayer_perceptron(x)

with tf.variable_scope("regression"):
    y_reg, regression_variables = model.regression(x)

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder(tf.float32)
    y_conv, conv_variables = model.convolutional(x, keep_prob)

with tf.variable_scope("rnn"):
    y_rnn, _ = model.rnn_network(x)
rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='rnn')
# train
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy_reg = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_reg))
cross_entropy_conv = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
cross_entropy_percep =  tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_percep))
cross_entropy_rnn = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_rnn))

train_reg = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_reg)
train_conv = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_conv)
train_percep = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_percep)
train_rnn = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_rnn)

correct_prediction_reg = tf.equal(tf.argmax(y_, 1), tf.argmax(y_reg, 1))
correct_prediction_conv = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
correct_prediction_percep = tf.equal(tf.argmax(y_, 1), tf.argmax(y_percep, 1))
correct_prediction_rnn = tf.equal(tf.argmax(y_, 1), tf.argmax(y_rnn, 1))

accuracy_reg = tf.reduce_mean(tf.cast(correct_prediction_reg, tf.float32))
accuracy_conv = tf.reduce_mean(tf.cast(correct_prediction_conv, tf.float32))
accuracy_percep = tf.reduce_mean(tf.cast(correct_prediction_percep, tf.float32))
accuracy_rnn = tf.reduce_mean(tf.cast(correct_prediction_rnn, tf.float32))

saver = tf.train.Saver(conv_variables+perceptron_variables+regression_variables+rnn_variables)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(N_TRAIN_STEPS):
        batch = data.train.next_batch(BATCH_SIZE)

        sess.run([train_reg, train_conv ,train_rnn, train_percep],
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})

        if i % 100 == 0:
            batch = data.test.next_batch(BATCH_SIZE/2)
            reg_acc, conv_acc, per_acc, rnn_acc = sess.run([accuracy_reg, accuracy_conv, accuracy_percep, accuracy_rnn],
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("iteration : {} Accuracy of \n Regression : {}, convolution : {}, Perceptron : {}, RNN  {}".format(i, reg_acc, conv_acc, per_acc, rnn_acc))
            saver.save(
                sess, os.path.join(os.path.dirname(__file__), 'data', 'mnist.ckpt'),global_step=i,
                write_meta_graph=False, write_state=False)

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'mnist.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)
