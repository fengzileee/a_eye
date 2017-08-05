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

def conv_net(X, keep_prob):
    def lrelu(x, a):
        return tf.nn.relu(x) - a * tf.nn.relu(-x)

    def conv_unit(index, input, filter_size,  
            pool_size, in_depth, out_depth, padding = 'SAME', 
            leakage = 0.1, isPool = False):
        conv_w = tf.Variable(tf.truncated_normal(
                shape = (filter_size, filter_size, in_depth, out_depth), 
                mean = 0, stddev = 0.1), 
            name = 'conv' + str(index) + '_w')
        conv_b = tf.Variable(-0.4 * tf.ones(out_depth), 
            name = 'conv' + str(index) + '_b')

        conv_out = tf.nn.conv2d(input = input, 
                filter = conv_w, strides = [1,1,1,1], 
                padding = padding) 
        conv_out = tf.nn.bias_add(conv_out, conv_b)
        conv_out = lrelu(conv_out, leakage)

        if isPool:
            pool_out = tf.nn.max_pool(value = conv_out, 
                    ksize = [1, pool_size, pool_size, 1], 
                    strides = [1, pool_size, pool_size, 1], 
                    padding = padding) 
            output = tf.nn.dropout(pool_out, keep_prob)
        else:
            output = tf.nn.dropout(conv_out, keep_prob)

        return output

    def full_unit(index, input, in_size, out_size, 
            leakage = 0.3):
        full_w = tf.Variable(tf.truncated_normal(shape = (in_size, out_size), 
                mean = 0, stddev = 0.1), 
                name = 'full'+str(index) + '_w')
        full_b = tf.Variable(tf.zeros(out_size), 
            name = 'full' + str(index) + '_b')

        full_out = tf.add(tf.matmul(input, full_w), full_b) 
        full_out = lrelu(full_out, leakage) 

        output = tf.nn.dropout(full_out, keep_prob)

        return output

    conv_out_depth = [None, 
            32, 64, 128, 64, 128, 
            256, 128, 256, 512, 256, 
            512, 256, 512, 1024, 512, 
            1024, 512, 1024, 2] 

    conv_filter_size = [None, 
            3, 3, 3, 1, 3,
            3, 1, 3, 3, 1,
            3, 1, 3, 3, 1,
            3, 1, 3, 1]

    pool_index = [1, 2, 5, 8]

    prev = conv_unit(index = 1, input = X, 
            filter_size = conv_filter_size[1], 
            pool_size = 2, 
            in_depth = 1, 
            out_depth = conv_out_depth[1], 
            isPool = True)

    for i in range (2, 20):
        prev = conv_unit(index = i, input = prev, 
                filter_size = conv_filter_size[i], 
                pool_size = 2, 
                in_depth = conv_out_depth[i-1], 
                out_depth = conv_out_depth[i], 
                isPool = i in pool_index)
        
    prev = tf.reduce_mean(prev, 1)
    prev = tf.reduce_mean(prev, 1)

    return prev



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

