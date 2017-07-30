import tensorflow as tf
from tensorflow.contrib.layers import flatten

def network(X, keep_probability):
    mu = 0;
    sigma = 0.1; 
    
    # layer 1, conv
    filter1_thickness = 32; 
    
    conv1_w = tf.Variable(tf.truncated_normal(shape = (5,5,1,filter1_thickness), 
                                             mean = mu, stddev = sigma), 
                         name = 'conv1_w')
    conv1_b = tf.Variable(tf.zeros(filter1_thickness), 
                         name = 'conv1_b')
    
    conv1 = tf.nn.bias_add(tf.nn.conv2d(input = X, filter = conv1_w, strides = [1,1,1,1], 
                        padding = 'VALID'),conv1_b)
    conv1 = tf.nn.relu(conv1) 
    conv1 = tf.nn.max_pool(value = conv1, ksize = [1,3,3,1], 
                           strides = [1,2,2,1], padding = 'VALID')
    
    # layer 2, conv 
    filter2_thickness = 32 
    
    conv2_w = tf.Variable(tf.truncated_normal(shape = (5,5,filter1_thickness,filter2_thickness), 
                                             mean = mu, stddev = sigma),
                         name = 'conv2_w')
    conv2_b = tf.Variable(tf.zeros(filter2_thickness), 
                         name = 'conv2_b')
    
    conv2 = tf.nn.bias_add(tf.nn.conv2d(input = conv1, filter = conv2_w, strides = [1,1,1,1], 
                        padding = 'VALID'),conv2_b)
    conv2 = tf.nn.relu(conv2) 
    conv2 = tf.nn.avg_pool(value = conv2, ksize = [1,3,3,1], 
                           strides = [1,2,2,1], padding = 'VALID')
    
    # layer 3
    filter3_thickness =  64
    
    conv3_w = tf.Variable(tf.truncated_normal(shape = (5,5,filter2_thickness,
        filter3_thickness), 
                                             mean = mu, stddev = sigma),
                         name = 'conv3_w')
    conv3_b = tf.Variable(tf.zeros(filter3_thickness), 
                         name = 'conv3_b')
    
    conv3 = tf.nn.bias_add(tf.nn.conv2d(input = conv2, filter = conv3_w, strides = [1,1,1,1], 
                        padding = 'VALID'),conv3_b)
    conv3 = tf.nn.relu(conv3) 
    conv3 = tf.nn.avg_pool(value = conv3, ksize = [1,3,3,1], 
                           strides = [1,2,2,1], padding = 'VALID')
    conv3 = flatten(conv3)
    #conv3 = tf.nn.dropout(conv5, keep_probability)
    layer3_out = 576

    # layer 4, full
    layer4_out = 64
    full4_w = tf.Variable(tf.truncated_normal(shape = (layer3_out,layer4_out), 
                                             mean = mu, stddev = sigma), 
                          name = 'full4_w')
    full4_b = tf.Variable(tf.zeros(layer4_out), 
                         name = 'full4_b')
    
    full4 = tf.add(tf.matmul(conv3, full4_w), full4_b)
    full4 = tf.nn.relu(full4)
       
    # layer 5, full, final
    full5_w = tf.Variable(tf.truncated_normal(shape = (layer4_out,2), 
                                             mean = mu, stddev = sigma), 
                          name = 'full5_w')
    full5_b = tf.Variable(tf.zeros(2), 
                         name = 'full5_b')
    
    full5 = tf.add(tf.matmul(full4, full5_w), full5_b)
    logits = tf.nn.relu(full5)
    
    return logits
