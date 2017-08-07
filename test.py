import simplenet
from aeye import *


X_ph, Y_ph, keep_p_ph, training_operation, accuracy_operation,\
        predict_operation, out\
        = construct_graph(network_builder = simplenet.conv_net, 
                learning_rate = 0.00001)
saver = tf.train.Saver()

split_indeces = [1, 2, 3, 4, 5]
mag = 100  # 40 100 200 400

for split_index in split_indeces:
    test_data = './BreaKHis_data/split' + str(split_index) + '/' + str(mag) + 'X_test.txt'

    model_name = 'model_cnn_' + str(split_index) + '.ckpt'
    model_dir = './model'
    model_file = model_dir + '/' + model_name
    
    output_file_name = './output/test' + str(split_index) + '.out'
    print('', end="", file = open(output_file_name, 'w'))
    fout = open(output_file_name, 'a')
    
    X_test_adr, Y_test = parse_txt(test_data)
    
    BATCH_SIZE = 1

    [accuracy, loss] = evaluate(X = X_test_adr, y = Y_test, 
         model = model_file, batch_size = BATCH_SIZE, 
         accuracy_operation = accuracy_operation, 
         out = out, X_ph = X_ph, Y_ph = Y_ph, 
         keep_p_ph = keep_p_ph, saver = saver)
     
    print('Accuracy: {:.3f}'.format(accuracy))
    print('Loss: {:.6f}'.format(loss))
    print('Accuracy: {:.3f}'.format(accuracy), 
                 file = fout)
    print('Loss: {:.6f}'.format(loss), 
                 file = fout)

    fout.close()
