import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
# Softmax Regression Model
def regression(x):
    W = tf.Variable(tf.zeros([784, 10]), name="r_W")
    b = tf.Variable(tf.zeros([10]), name="r_b")
    y = tf.matmul(x, W) + b
    return y, [W, b]

# Multilayer Perceptron model
def multilayer_perceptron(x):
    n_input_layer = 784
    n_class = 10
    n_h1 = 256
    n_h2 = 256
    W1 = tf.Variable(tf.random_normal([n_input_layer, n_h1]), name="pW1")
    b1 = tf.Variable(tf.random_normal([n_h1]), name="pb1")
    y1 = tf.add( tf.matmul(x, W1) , b1 )

    W2 = tf.Variable(tf.random_normal([n_h1, n_h2]), name="pW2")
    b2 = tf.Variable(tf.random_normal([n_h2]), name="pb2")
    y2 = tf.add( tf.matmul(y1, W2) , b2 )

    W3 = tf.Variable(tf.random_normal([n_h2, n_class]), name="pW3")
    b3 = tf.Variable(tf.random_normal([n_class]), name="pb3")
    y3 = tf.add( tf.matmul(y2, W3) , b3)

    return y3, [W1, b1, W2, b2, W3, b3]

# Multilayer Convolutional Network
def convolutional(x, keep_prob):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # First Convolutional Layer
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
def rnn_network(x):
    n_classes = 10
    chunk_size = 28
    n_chunks = 28
    rnn_size = 128
    x = tf.reshape(x, [-1,n_chunks,chunk_size])
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output, [layer['weights'],  layer['biases'] ]+list(states)
