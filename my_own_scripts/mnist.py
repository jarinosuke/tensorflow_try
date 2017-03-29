import tensorflow as tf

# from https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/mnist/input_data.py
import input_data

import time

start_time = time.time()
print "start time:" + str(start_time)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set training images
# 784 is pixel range of 28x28
x = tf.placeholder(tf.float32, [None, 784])

# weight
W = tf.Variable(tf.zeros([784, 10]))

# bias
b = tf.Variable(tf.zeros([10]))

# soft-max regression
y = tf.nn.softmax(tf.matmul(x, W) + b)

# calculate cross_entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y * tf.log(y))

# use gradient-descent to minimize cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# start session
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

# train 1000 times

for i in range(1000):
    # choose 100 set randomly
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

print "done training"

# prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

end_time = time.time()
print "end time:" + str(end_time)
print "spent time:" +str(end_time - start_time)
