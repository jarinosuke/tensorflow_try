import numpy as np
import tensorflow as tf

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

num_inputs = 20
num_classes = 1

with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, num_inputs], name="x-input")
    y = tf.placeholder(tf.float32, [None, num_classes], name="y-input")

with tf.name_scope("model"):
    W = tf.Variable(tf.zeros([num_inputs, num_classes]), name="W") #weight
    b = tf.Variable(tf.zeros([num_classes]), name="b") #bias

y_pred = tf.sigmoid(tf.matmul(x, W) + b)

with tf.name_scope("hyperparameters"):
    regularization = tf.placeholder(tf.float32, name="regularization")
    learning_rate = tf.placeholder(tf.float32, name="learning-rate")

with tf.name_scope("loss-function"):
    loss = tf.losses.log_loss(labels=y, predictions=y_pred)
    loss += regularization * tf.nn.l2_loss(W)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope("score"):
    correct_prediction = tf.equal(tf.to_float(y_pred > 0.5), y)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name="accuracy")

with tf.name_scope("inference"):
    inference = tf.to_float(y_pred > 0.5, name="inference")
