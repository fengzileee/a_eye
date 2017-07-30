import tensorflow as tf
from tensorflow.contrib.layers import flatten

def network(X, keep_probability):
    mu = 0;
    sigma = 0.1; 
    
    # conv 1
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
    
    # conv 2, dropout
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
    conv2 = tf.nn.dropout(conv2, keep_probability)
    
    # conv 3
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
    conv3 = tf.nn.max_pool(value = conv3, ksize = [1,3,3,1], 
                           strides = [1,2,2,1], padding = 'VALID')

    # conv 4, dropout
    filter4_thickness = 64

    conv4_w = tf.Variable(tf.truncated_normal(shape = (5,5,filter3_thickness,
        filter4_thickness), 
                                             mean = mu, stddev = sigma),
                         name = 'conv4_w')
    conv4_b = tf.Variable(tf.zeros(filter4_thickness), 
                         name = 'conv4_b')
    
    conv4 = tf.nn.bias_add(tf.nn.conv2d(input = conv3, filter = conv4_w, strides = [1,1,1,1], 
                        padding = 'VALID'),conv4_b)
    conv4 = tf.nn.relu(conv4) 
    conv4 = tf.nn.avg_pool(value = conv4, ksize = [1,3,3,1], 
                           strides = [1,2,2,1], padding = 'VALID')
    conv4 = tf.nn.dropout(conv4, keep_probability)

    # conv 5
    filter5_thickness = 128

    conv5_w = tf.Variable(tf.truncated_normal(shape = (5,5,filter4_thickness,
        filter5_thickness), 
                                             mean = mu, stddev = sigma),
                         name = 'conv5_w')
    conv5_b = tf.Variable(tf.zeros(filter5_thickness), 
                         name = 'conv5_b')
    
    conv5 = tf.nn.bias_add(tf.nn.conv2d(input = conv4, filter = conv5_w, strides = [1,1,1,1], 
                        padding = 'VALID'),conv5_b)
    conv5 = tf.nn.relu(conv5) 
    conv5 = tf.nn.avg_pool(value = conv5, ksize = [1,3,3,1], 
                           strides = [1,2,2,1], padding = 'VALID')


    conv_out = flatten(conv5)
    conv_flatten_out = 1536

    # full 1
    full1_out = 512 
    full1_w = tf.Variable(tf.truncated_normal(shape = (conv_flatten_out,full1_out), 
                                             mean = mu, stddev = sigma), 
                          name = 'full1_w')
    full1_b = tf.Variable(tf.zeros(full1_out), 
                         name = 'full1_b')
    
    full1 = tf.add(tf.matmul(conv_out, full1_w), full1_b)
    full1 = tf.nn.relu(full1)
 
    # full 2
    full2_out = 64 
    full2_w = tf.Variable(tf.truncated_normal(shape = (full1_out,full2_out), 
                                             mean = mu, stddev = sigma), 
                          name = 'full2_w')
    full2_b = tf.Variable(tf.zeros(full2_out), 
                         name = 'full2_b')
    
    full2 = tf.add(tf.matmul(full1, full2_w), full2_b)
    full2 = tf.nn.relu(full2)
    full2 = tf.nn.dropout(full2, keep_probability)
       
    # full out, final
    fullo_w = tf.Variable(tf.truncated_normal(shape = (full2_out,2), 
                                             mean = mu, stddev = sigma), 
                          name = 'fullo_w')
    fullo_b = tf.Variable(tf.zeros(2), 
                         name = 'fullo_b')
    
    fullo = tf.add(tf.matmul(full2, fullo_w), fullo_b)
    logits = tf.nn.relu(fullo)
    
   
    return logits
