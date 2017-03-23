import os
import numpy as np
import tensorflow as tf

checkpoint_dir = "../tmp/"
print_every = 1000
save_every = 10000

X_train = np.load(checkpoint_dir + "X_train.npy")
y_train = np.load(checkpoint_dir + "y_train.npy")

num_inputs = 20
num_classes = 1

# setup computational graph

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

# training classifier

init = tf.global_variables_initializer()

# For writing training checkpoints and reading them back in.
saver = tf.train.Saver()
tf.gfile.MakeDirs(checkpoint_dir)

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, checkpoint_dir, "graph.rb", False)

    sess.run(init)

    # Sanity check: the initial loss should be 0.693146, which is -ln(0.5).
    loss_value = sess.run(loss, feed_dict={x: X_train, y: y_train, regularization: 0})
    print("Initial loss:", loss_value)

    step = 0
    while True:
        perm = np.arange(len(X_train))
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]

        feed = {x: X_train, y: y_train, learning_rate: 1e-2, 
                                regularization: 1e-5}
        sess.run(train_op, feed_dict=feed)

        if step % print_every == 0:
           train_accuracy, loss_value = sess.run([accuracy, loss], feed_dict=feed)
           print("step: %4d, loss: %.4f, training accuracy: %.4f" % 
                   (step, loss_value, train_accuracy))
        
        step += 1

        if step % save_every == 0:
            checkpoint_file = os.path.join(checkpoint_dir, "model")
            saver.save(sess, checkpoint_file) 
            print("*** SAVED MODEL ***")
