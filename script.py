import simplenet
from aeye import *

model_name = 'model_trial.ckpt'
model_dir = './model_trial'
model_file = model_dir + '/' + model_name

print("construct graph")
X_ph, Y_ph, keep_p_ph, training_operation, accuracy_operation,\
        predict_operation, out\
        = construct_graph(network_builder = simplenet.network, learning_rate = 0.000001)
saver = tf.train.Saver()
print("fnished construction of graph")

training_data = './BreaKHis_data/split1/200X_train.txt'
validation_data = './BreaKHis_data/split1/200X_val.txt'
test_data = './BreaKHis_data/split1/200X_test.txt'

print("parse data")
output_file_name = 'split1.out'
print('', end="", file = open(output_file_name, 'w'))
fout = open(output_file_name, 'a')

X_train_adr, Y_train = parse_txt(training_data)
X_val_adr, Y_val = parse_txt(validation_data)
X_test_adr, Y_test = parse_txt(test_data)
print("finished parsing data")

print("start training")
EPOCH_NO = 100
BATCH_SIZE = 1

for epoch in range(EPOCH_NO):
    print('========= Epoch {} started ... ========='.format(epoch + 1))
    print('========= Epoch {} started ... ========='.format(epoch + 1), file = fout)
    train(X = X_train_adr, y = Y_train, model = model_file,
            batch_size = BATCH_SIZE, keep_prob = 1, training_operation = training_operation, 
            X_ph = X_ph, Y_ph = Y_ph, keep_p_ph = keep_p_ph, saver = saver, out = out)

    [accuracy, loss] = evaluate(X = X_val_adr, y = Y_val, model = model_file,
            batch_size = BATCH_SIZE, accuracy_operation = accuracy_operation, out = out,
            X_ph = X_ph, Y_ph = Y_ph, keep_p_ph = keep_p_ph, saver = saver)

    print('Epoch {}/{}, Validation accuracy: {:.3f}'.format(epoch+1, EPOCH_NO, accuracy))
    print('Epoch {}/{}, Loss: {:.6f}'.format(epoch+1, EPOCH_NO, loss))
    print('Epoch {}/{}, Validation accuracy: {:.3f}'.format(epoch+1, EPOCH_NO, accuracy), 
            file = fout)
    print('Epoch {}/{}, Loss: {:.6f}'.format(epoch+1, EPOCH_NO, loss), 
            file = fout)
print("finished training")
