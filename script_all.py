import simplenet
from aeye import *


X_ph, Y_ph, keep_p_ph, training_operation, accuracy_operation,\
        predict_operation, out\
        = construct_graph(network_builder = simplenet.conv_net, 
                learning_rate = 0.00001)
saver = tf.train.Saver()

split_indeces = [1, 2, 3, 4, 5]
mag = 100  # 40 100 200 400

X_train_adr = []
Y_train = []
X_val_adr = []
Y_val = []
X_test_adr = []
Y_test = []

for split_index in split_indeces:
    training_data = './BreaKHis_data/split' + str(split_index) + '/' + str(mag) + 'X_train.txt'
    validation_data = './BreaKHis_data/split' + str(split_index) + '/' + str(mag) + 'X_val.txt'
    test_data = './BreaKHis_data/split' + str(split_index) + '/' + str(mag) + 'X_test.txt'
    
    X_train_adr_tmp, Y_train_tmp = parse_txt(training_data)
    X_val_adr_tmp, Y_val_tmp = parse_txt(validation_data)
    X_test_adr_tmp, Y_test_tmp = parse_txt(test_data)
    
    X_train_adr = X_train_adr + X_train_adr_tmp
    Y_train = Y_train + Y_train_tmp
    X_val_adr = X_val_adr + X_val_adr_tmp
    Y_val = Y_val + Y_val_tmp
    X_test_adr = X_test_adr + X_test_adr_tmp
    Y_test = Y_test + Y_test_tmp

model_name = 'model_cnn_all.ckpt'
model_dir = './model'
model_file = model_dir + '/' + model_name

output_file_name = 'all.out'
print('', end="", file = open(output_file_name, 'w'))
fout = open(output_file_name, 'a')

print("start training")
EPOCH_NO = 10
BATCH_SIZE = 1

for epoch in range(EPOCH_NO):
    print('========= Epoch {} started ... ========='.format(epoch + 1))
    print('========= Epoch {} started ... ========='.format(epoch + 1), file = fout)
    train(X = X_train_adr, y = Y_train, model = model_file,
    	batch_size = BATCH_SIZE, keep_prob = 1, 
    	training_operation = training_operation, 
    	X_ph = X_ph, Y_ph = Y_ph, keep_p_ph = keep_p_ph, 
    	saver = saver, out = out)
    
    [accuracy, loss] = evaluate(X = X_val_adr, y = Y_val, 
    	model = model_file, batch_size = BATCH_SIZE, 
    	accuracy_operation = accuracy_operation, 
    	out = out, X_ph = X_ph, Y_ph = Y_ph, 
    	keep_p_ph = keep_p_ph, saver = saver)
    
    print('Epoch {}/{}, Validation accuracy: {:.3f}'.format(epoch+1, EPOCH_NO, accuracy))
    print('Epoch {}/{}, Loss: {:.6f}'.format(epoch+1, EPOCH_NO, loss))
    print('Epoch {}/{}, Validation accuracy: {:.3f}'.format(epoch+1, EPOCH_NO, accuracy), 
    	file = fout)
    print('Epoch {}/{}, Loss: {:.6f}'.format(epoch+1, EPOCH_NO, loss), 
    	file = fout)
fout.close()
