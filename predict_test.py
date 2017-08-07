import simplenet
from aeye import *

split_index = 3
model_name = 'model_cnn_' + str(split_index) + '.ckpt'
model_dir = './model'
model_file = model_dir + '/' + model_name

X_ph, Y_ph, keep_p_ph, training_operation, accuracy_operation,\
        predict_operation, out\
        = construct_graph(network_builder = simplenet.conv_net, 
                learning_rate = 0.00001)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, model_file)

# GUI comes in here

x_adr = './BreaKHis_data/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-16875/100X/SOB_M_DC-14-16875-100-004.png'
#y = 1

result = predict(x_adr, predict_operation, \
        X_ph, keep_p_ph, sess)
print(result)


sess.close()
