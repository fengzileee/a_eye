import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

def network(X, keep_probability):
    
    # ======= Variables ========
    mu = 0;
    sigma = 0.1; 

    # conv 1, var
    filter1_thickness = 32; 
    
    conv1_w = tf.Variable(tf.truncated_normal(shape = (5,5,1,filter1_thickness), 
                                             mean = mu, stddev = sigma), 
                         name = 'conv1_w')
    conv1_b = tf.Variable(tf.zeros(filter1_thickness), 
                         name = 'conv1_b')
    
    # conv 2, var
    filter2_thickness = 32 
    
    conv2_w = tf.Variable(tf.truncated_normal(shape = (5,5,
        filter1_thickness,
        filter2_thickness), 
        mean = mu, stddev = sigma),
                         name = 'conv2_w')
    conv2_b = tf.Variable(tf.zeros(filter2_thickness), 
                         name = 'conv2_b')
    
    # conv 3, var
    filter3_thickness =  64
    
    conv3_w = tf.Variable(tf.truncated_normal(shape = (5,5,
        filter2_thickness,
        filter3_thickness), 
        mean = mu, stddev = sigma),
        name = 'conv3_w')
    conv3_b = tf.Variable(tf.zeros(filter3_thickness), 
                         name = 'conv3_b')

    # full 1, var
    conv_flatten_out = 576
    full1_out = 64 
    full1_w = tf.Variable(tf.truncated_normal(shape = (conv_flatten_out,
        full1_out), mean = mu, stddev = sigma), name = 'full1_w')
    full1_b = tf.Variable(tf.zeros(full1_out), 
                         name = 'full1_b')

    # full o, var
    fullo_w = tf.Variable(tf.truncated_normal(shape = (full1_out,2), 
                                             mean = mu, stddev = sigma), 
                          name = 'fullo_w')
    fullo_b = tf.Variable(tf.zeros(2), 
                         name = 'fullo_b')

    # ============= forward ==============
    PATCH_SIZE = 64;
    PATCH_STRIDE = PATCH_SIZE/2;
    PATCH_NO = np.floor((350 - PATCH_STRIDE) / PATCH_STRIDE) * \
        np.floor((230 - PATCH_STRIDE) / PATCH_STRIDE) 
    PATCH_NO = tf.cast(PATCH_NO, tf.int32)

    # patch
    patches = tf.extract_image_patches(images = X, 
            ksizes = [1, PATCH_SIZE, PATCH_SIZE, 1], 
            strides = [1, PATCH_STRIDE, PATCH_STRIDE, 1], 
            rates = [1, 1, 1, 1],
            padding = 'VALID')

    patches = tf.reshape(tensor = patches, 
            shape = [-1, PATCH_NO, PATCH_SIZE, PATCH_SIZE])
    
    # rearrange patches
    patches_t = tf.transpose(patches, perm = [1, 0, 2, 3])

    def forward(p):
        p = tf.reshape(tensor = p, 
                shape = [-1, PATCH_SIZE, PATCH_SIZE, 1])

        conv_out = flatten(p)
        # conv 1
        conv1 = tf.nn.bias_add(tf.nn.conv2d(input = p, 
            filter = conv1_w, 
            strides = [1,1,1,1], 
            padding = 'VALID'),conv1_b)
        conv1 = tf.nn.relu(conv1) 
        conv1 = tf.nn.max_pool(value = conv1, ksize = [1,3,3,1], 
                               strides = [1,2,2,1], padding = 'VALID')

        # conv 2, dropout
        conv2 = tf.nn.bias_add(tf.nn.conv2d(input = conv1, 
            filter = conv2_w, 
            strides = [1,1,1,1], 
            padding = 'VALID'), conv2_b)
        conv2 = tf.nn.relu(conv2) 
        conv2 = tf.nn.avg_pool(value = conv2, ksize = [1,3,3,1], 
                               strides = [1,2,2,1], padding = 'VALID')
        conv2 = tf.nn.dropout(conv2, keep_probability)

        # conv 3
        conv3 = tf.nn.bias_add(tf.nn.conv2d(input = conv2, 
            filter = conv3_w, 
            strides = [1,1,1,1], 
            padding = 'VALID'),conv3_b)
        conv3 = tf.nn.relu(conv3) 
        conv3 = tf.nn.avg_pool(value = conv3, ksize = [1,3,3,1], 
                               strides = [1,2,2,1], padding = 'VALID')

        conv_out = flatten(conv3)

        # full 1
        full1 = tf.add(tf.matmul(conv_out, full1_w), full1_b)
        full1 = tf.nn.relu(full1)
          
        # full out, final
        
        fullo = tf.add(tf.matmul(full1, fullo_w), fullo_b)

        return fullo
        
    # process patch
    patch_logits = tf.map_fn(forward, patches_t)
    logits = tf.reduce_sum(input_tensor = patch_logits, 
            axis = 0)

   
    return logits

def stupid_net(X, keep_prob):
    mu = 0;
    sigma = 0.1; 

    X = flatten(X)
    full1_out = 128 
    full1_w = tf.Variable(tf.truncated_normal(shape = (80500 ,full1_out), 
                                             mean = mu, stddev = sigma), 
                          name = 'full1_w')
    full1_b = tf.Variable(tf.zeros(full1_out), 
                         name = 'full1_b')
    
    full1 = tf.add(tf.matmul(X, full1_w), full1_b)
    full1 = tf.nn.sigmoid(full1)
 
    # full 2
    full2_out = 2 
    full2_w = tf.Variable(tf.truncated_normal(shape = (full1_out,full2_out), 
                                             mean = mu, stddev = sigma), 
                          name = 'full2_w')
    full2_b = tf.Variable(tf.zeros(full2_out), 
                         name = 'full2_b')
    
    full2 = tf.add(tf.matmul(full1, full2_w), full2_b)
    logits = tf.nn.sigmoid(full2)

    return logits

