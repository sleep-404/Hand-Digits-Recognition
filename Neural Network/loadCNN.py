# Imports
import tensorflow as tf
import numpy as np
import os
 
from defines import * 
from layers import *
from plotting import *


graph = tf.Graph()
with graph.as_default():
    # TF Placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, width, height, 1], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name="y")
    keep_prob_conv_tf = tf.placeholder_with_default(1.0, shape=(), name="KEEP_CONV")
    keep_prob_fc_tf = tf.placeholder_with_default(1.0, shape=(), name="KEEP_FC")

    # Weights
    W_conv1_b = tf.Variable(tf.truncated_normal([7, 7, 1, 32], stddev=stddev), name="W_conv1_b")
    W_conv2_b = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=stddev), name="W_conv2_b")
    W_conv3_b = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=stddev), name="W_conv3_b")
    W_conv4_b = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=stddev), name="W_conv4_b")
    W_fc_b = tf.Variable(tf.truncated_normal([7 * 7 * 64, dense_size], stddev=stddev), name="W_fc_b")
    W_out_b = tf.Variable(tf.truncated_normal([dense_size, classes], stddev=stddev), name="W_out_b")

    b_conv1_b = tf.Variable(tf.constant(bias_weight_init, shape=[32]), name="b_conv1_b")
    b_conv2_b = tf.Variable(tf.constant(bias_weight_init, shape=[64]), name="b_conv2_b")
    b_conv3_b = tf.Variable(tf.constant(bias_weight_init, shape=[64]), name="b_conv3_b")
    b_conv4_b = tf.Variable(tf.constant(bias_weight_init, shape=[64]), name="b_conv4_b")
    b_fc_b = tf.Variable(tf.constant(bias_weight_init, shape=[dense_size]), name="b_fc_b")
    b_out_b = tf.Variable(tf.constant(bias_weight_init, shape=[classes]), name="b_out_b")

    # Saver object
    new_saver = tf.train.Saver({"W_conv1": W_conv1_b, "W_conv2": W_conv2_b, "W_conv3": W_conv3_b, "W_conv4": W_conv4_b,
                                "b_conv1": b_conv1_b, "b_conv2": b_conv2_b, "b_conv3": b_conv3_b, "b_conv4": b_conv4_b,
                                "W_fc": W_fc_b, "W_out": W_out_b, "b_fc": b_fc_b, "b_out": b_out_b})

    # Define the Deep Neural Network (CNN)
    def nn_model(x, keep_prob_conv_tf, keep_prob_fc_tf, graph):
        with graph.as_default():
            conv1_layer  = {"weights": W_conv1_b, "biases": b_conv1_b}
            conv2_layer  = {"weights": W_conv2_b, "biases": b_conv2_b}
            conv3_layer  = {"weights": W_conv3_b, "biases": b_conv3_b}
            conv4_layer  = {"weights": W_conv4_b, "biases": b_conv4_b}
            fc_layer     = {"weights": W_fc_b, "biases": b_fc_b}
            output_layer = {"weights": W_out_b, "biases": b_out_b}
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

def nn_predict(image=None):
    with tf.Session(graph=graph) as sess:
        # Load trained model
        new_saver = tf.train.import_meta_graph(dir_path + "/data/model.ckpt.meta")
        new_saver.restore(sess, dir_path + "/data/model.ckpt")
        # Feed image to network and get prediction
        pred = nn_model(image.reshape((-1, 28, 28, 1)), keep_prob_conv_tf, keep_prob_fc_tf, graph)
        pred = sess.run(tf.argmax(tf.nn.softmax(pred), 1))
        return pred

if __name__ == "__main__":
    nn_predict()