from flask import Flask, jsonify, render_template, request
import numpy as np
import tensorflow as tf


from mnist import model


x = tf.placeholder("float", [None, 784])
sess = tf.Session()

# restore trained data
with tf.variable_scope("perceptron"):
    y_percep, perceptron_variables = model.multilayer_perceptron(x)

with tf.variable_scope("regression"):
    y_reg, regression_variables = model.regression(x)

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder(tf.float32)
    y_conv, conv_variables = model.convolutional(x, keep_prob)
with tf.variable_scope("rnn"):
    y_rnn,_ = model.rnn_network(x)
rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='rnn')
saver = tf.train.Saver(conv_variables+perceptron_variables+regression_variables+rnn_variables)
saver.restore(sess, "mnist/data/mnist.ckpt")


def regression(input):
    return sess.run(tf.nn.softmax(y_reg), feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(tf.nn.softmax(y_conv), feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

def perceptron(input):
    return sess.run(tf.nn.softmax(y_percep), feed_dict={x: input}).flatten().tolist()

def rnn(input):
    return sess.run(tf.nn.softmax(y_rnn), feed_dict={x: input}).flatten().tolist()


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    output3 = perceptron(input)
    output4 = rnn(input)
    return jsonify(results=[output1, output2, output3, output4])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
