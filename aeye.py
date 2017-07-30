import matplotlib.image as mimg
import cv2
import tensorflow as tf
import os.path
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def norm_gray(image):
    low = 0.1
    high = 0.9 
    min_v = image.min()
    max_v = image.max()
    return low + (image - min_v) / (max_v - min_v) * (high - low)

def preproctf(data_set):
    pro_data_set = []
    for img in data_set:
        img = grayscale(img)
        img = norm_gray(img)
        img = cv2.resize(img,(350, 230))
        pro_data_set.append(img)
    pro_data_set = np.reshape(pro_data_set,(len(pro_data_set),350,230,1))
    return pro_data_set

def evaluate(X,y,model,batch_size, accuracy_operation, X_ph, Y_ph, keep_p_ph, saver):
    total_eval = len(X)
    total_accuracy = 0 
    with tf.Session() as sess:
        saver.restore(sess,model)
        for offset in range(0, total_eval, batch_size):
            x_batch, y_batch = X[offset:offset + batch_size], y[offset:offset + batch_size]
            x_batch = load_image(x_batch)
            x_batch = preproctf(x_batch)
            accuracy = sess.run(accuracy_operation, 
                                feed_dict = {X_ph:x_batch, Y_ph:y_batch, keep_p_ph:1})
            total_accuracy += (accuracy * len(x_batch))
    return (total_accuracy/total_eval)

def initialize(model,saver):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, model)

def train(X,y,model,batch_size, keep_prob, training_operation, X_ph, Y_ph, keep_p_ph, saver):
    if not os.path.isfile(model+'.meta'):
        initialize(model, saver)
        print('initialized.')
    with tf.Session() as sess: 
        saver.restore(sess,model)
        num_train = len(X)
        X, y = shuffle(X, y)
        for offset in range(0,num_train,batch_size): 
            x_batch, y_batch = X[offset:offset + batch_size], y[offset:offset + batch_size]
            x_batch = load_image(x_batch)
            x_batch = preproctf(x_batch)
            sess.run(training_operation, feed_dict = {X_ph:x_batch, Y_ph:y_batch, keep_p_ph:keep_prob})
        saver.save(sess, model)

def predict(X, model, predict_operation, saver):
    with tf.Session() as sess:
        saver.restore(sess,model)
        X = load_image(X)
        X = preproctf(X)
        ret = sess.run(predict_operation,feed_dict = {X_ph:X, keep_p: 1})
    return ret

def construct_graph(network_builder, learning_rate): 
    """
    network_builder(X_placeholder, keep_prob_placeholder) returns the logit
    input size: 350 x 230 x 1
    output size: 2

    learning_rate: float, learning rate
    """
    # placeholders
    X_ph = tf.placeholder(dtype = tf.float32,
            shape = (None, 350, 230, 1), 
            name = 'X_placeholder')
    Y_ph = tf.placeholder(dtype = tf.int32, 
            shape = (None), 
            name = 'Y_placeholder')
    keep_p_ph = tf.placeholder(dtype = tf.float32, 
            name = 'keep_prob_placeholder')

    # encode labels
    y_encoded = tf.one_hot(indices = Y_ph, depth = 2)

    # network output
    logits = network_builder(X_ph, keep_p_ph)

    # training pipeline
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_encoded)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss_operation)

    # evaluation pipeline
    isCorrect = tf.equal(tf.argmax(logits,1), tf.arg_max(y_encoded,1))
    accuracy_operation = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

    # prediction pipeline 
    predict_operation = tf.nn.top_k(tf.nn.softmax(logits,2)) # softmax of logits

    return X_ph, Y_ph, keep_p_ph, training_operation, accuracy_operation, predict_operation


# ============== data loading ==============
def parse_txt(filename):
    with open(filename, 'r') as f:
        X_adr = [] 
        Y = []
        for line in f:
            line = line.rstrip('\n')
            splited = line.split(' ')
            X_adr.append(splited[0])
            Y.append(splited[1])
    return X_adr, Y

def load_image(files_adr):
    """ 
    Takes a list of addresses given in the text file in split directory
    Return a list of numpy arrays, each representing one imega
    """
    ret = []
    for adr in files_adr:
        im = mimg.imread('./BreaKHis_data/' + adr)
        ret.append(im)
    ret = np.array(ret)
    return ret

