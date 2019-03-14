# Imports
import tensorflow as tf
import numpy as np
import os
 
from defines import *
from layers import *
from plotting import *
from data import *

# Train the Neural Network
def nn_train(hparam="Model"):
    ##############################################################################
    #                                 TF MODE                                   #
    ##############################################################################
    graph = tf.Graph()
 
    with graph.as_default():
        # TF Placeholders
        x = tf.placeholder(dtype=tf.float32, shape=[None, width, height, 1], name="x")
        y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name="y")
        keep_prob_conv_tf = tf.placeholder_with_default(1.0, shape=(), name="KEEP_CONV")
        keep_prob_fc_tf = tf.placeholder_with_default(1.0, shape=(), name="KEEP_FC")

        # Weights
        W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 1, 32], stddev=stddev), name="W_conv1")
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=stddev), name="W_conv2")
        W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=stddev), name="W_conv3")
        W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=stddev), name="W_conv4")
        W_fc = tf.Variable(tf.truncated_normal([7 * 7 * 64, dense_size], stddev=stddev), name="W_fc")
        W_out = tf.Variable(tf.truncated_normal([dense_size, classes], stddev=stddev), name="W_out")

        b_conv1 = tf.Variable(tf.constant(bias_weight_init, shape=[32]), name="b_conv1")
        b_conv2 = tf.Variable(tf.constant(bias_weight_init, shape=[64]), name="b_conv2")
        b_conv3 = tf.Variable(tf.constant(bias_weight_init, shape=[64]), name="b_conv3")
        b_conv4 = tf.Variable(tf.constant(bias_weight_init, shape=[64]), name="b_conv4")
        b_fc = tf.Variable(tf.constant(bias_weight_init, shape=[dense_size]), name="b_fc")
        b_out = tf.Variable(tf.constant(bias_weight_init, shape=[classes]), name="b_out")

    # Define the Deep Neural Network (CNN)
    def nn_model(x, keep_prob_conv_tf, keep_prob_fc_tf, graph):
        with graph.as_default():
            conv1_layer  = {"weights": W_conv1, "biases": b_conv1}
            conv2_layer  = {"weights": W_conv2, "biases": b_conv2}
            conv3_layer  = {"weights": W_conv3, "biases": b_conv3}
            conv4_layer  = {"weights": W_conv4, "biases": b_conv4}
            fc_layer     = {"weights": W_fc, "biases": b_fc}
            output_layer = {"weights": W_out, "biases": b_out}
            # Conv and pool layers
            conv1 = conv2d(x, conv1_layer["weights"])
            conv1 = tf.add(conv1, conv1_layer["biases"])
            conv1 = tf.nn.relu(conv1)

            conv2 = conv2d(conv1, conv2_layer["weights"])
            conv2 = tf.add(conv2, conv2_layer["biases"])
            conv2 = tf.nn.relu(conv2)

            pool1 = max_pool(conv2)
            drop1 = tf.nn.dropout(pool1, keep_prob_conv_tf)

            conv3 = conv2d(drop1, conv3_layer["weights"])
            conv3 = tf.add(conv3, conv3_layer["biases"])
            conv3 = tf.nn.relu(conv3)
            
            conv4 = conv2d(conv3, conv4_layer["weights"])
            conv4 = tf.add(conv4, conv4_layer["biases"])
            conv4 = tf.nn.relu(conv4)
            
            pool2 = max_pool(conv4)
            drop2 = tf.nn.dropout(pool2, keep_prob_conv_tf)
            # FC layers
            dense = flatten(pool2)

            fc1 = tf.matmul(dense, fc_layer["weights"])
            fc1 = tf.add(fc1, fc_layer["biases"])
            fc1 = tf.nn.relu(fc1)

            drop3 = tf.nn.dropout(fc1, keep_prob_fc_tf)

            fc2 = tf.matmul(drop3, output_layer["weights"])
            fc2 = tf.add(fc2, output_layer["biases"])
            return fc2

    ##############################################################################
    #                                   TF OPS                                   #
    ##############################################################################
    with graph.as_default():
        with tf.name_scope('error'):
            pred_op = nn_model(x, keep_prob_conv_tf, keep_prob_fc_tf, graph)
            error_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_op, labels=y))

        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error_op)

        with tf.name_scope('accuracy'):
            correct_result_op = tf.equal(tf.argmax(pred_op, 1), tf.argmax(y, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_result_op , tf.float32))

        with tf.name_scope("confusion"):
            confusion_matrix_op = tf.confusion_matrix(tf.argmax(pred_op, 1), tf.argmax(y, 1))

    ##############################################################################
    #                                START TRAIN                                 #
    ##############################################################################
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # Training
        print("\n\nStarting: ", hparam, " with TrainSize: ", train_size, " and TestSize: ", test_size)
        for epoch in range(epochs):
            mnist_data.shuffle_train()
            train_acc, test_acc = 0.0, 0.0
            train_loss, test_loss = 0.0, 0.0
            # Train the weights
            for i in range(train_mini_batches):
                epoch_x, epoch_y = mnist_data.next_train_batch(train_batch_size)
                _, c = sess.run([train_op, error_op], feed_dict={x: epoch_x, y: epoch_y, 
                   keep_prob_conv_tf: keep_prob_conv, keep_prob_fc_tf: keep_prob_fc})
            # Check the performance of the train set
            for i in range(train_mini_batches):
                epoch_x, epoch_y = mnist_data.next_train_batch(train_batch_size)
                a, c = sess.run([accuracy_op, error_op], feed_dict={x: epoch_x, y: epoch_y})
                train_acc += a
                train_loss += c
            train_acc = train_acc / train_mini_batches
            # Check the performance of the test set
            for i in range(test_mini_batches):
                epoch_x, epoch_y = mnist_data.next_test_batch(test_batch_size)
                a, c = sess.run([accuracy_op, error_op], feed_dict={x: epoch_x, y: epoch_y})
                test_acc += a
                test_loss += c
            test_acc = test_acc / test_mini_batches
            print("Epoch: ", epoch+1, " of ", epochs, "- Train loss: ", round(train_loss, 3), 
                " Test loss: ", round(test_loss, 3), " Train Acc: ", round(train_acc, 3), 
                " Test Acc: ", round(test_acc, 3))
        # Testing
        test_acc = 0.0
        test_loss = 0.0
        print("\n\nFinal testing!")
        for i in range(test_mini_batches):
                epoch_x, epoch_y = mnist_data.next_test_batch(test_batch_size)
                a, c, m = sess.run([accuracy_op, error_op, confusion_matrix_op], 
                    feed_dict={x: epoch_x, y: epoch_y})
                test_acc += a
                test_loss += c
        test_acc = test_acc / test_mini_batches
        print("Test Accuracy:\t", test_acc)
        print("Test Loss:\t", test_loss)
        # Save trained model (weights)
        save_path = saver.save(sess, dir_path + "/data/model.ckpt")
        print('Model saved in file', save_path)

def main():
    nn_train()

if __name__ == "__main__":
    main() 