import simplenet
from aeye import *

model_name = 'model_trial.ckpt'
model_dir = './model_trial'
model_file = model_dir + '/' + model_name

X_ph, Y_ph, keep_p_ph, training_operation, accuracy_operation, predict_operation\
        = construct_graph(network_builder = simplenet.network, learning_rate = 0.01)

training_data = './BreaKHis_data/split1/400X_train.txt'
validation_data = './BreaKHis_data/split1/400X_val.txt'
test_data = './BreaKHis_data/split1/400X_test.txt'

output_file_name = 'split1.out'
print('', end="", file = open(output_file_name, 'a'))
fout = open(output_file_name, 'a')

X_train_adr, Y_train = parse_txt(training_data)
X_val_adr, Y_val = parse_txt(validation_data)
X_test_adr, Y_test = parse_txt(test_data)

EPOCH_NO = 20

for epoch in range(EPOCH_NO):
    print('Epoch {} started ...'.format(epoch + 1))
    print('Epoch {} started ...'.format(epoch + 1), file = fout)
    train(X = X_train_adr, y = Y_train, model = model_file,
            batch_size = 32, keep_prob = 0.75, training_operation = training_operation, 
            X_ph = X_ph, Y_ph = Y_ph, keep_p_ph = keep_p_ph)

    accuracy = evaluate(X = X_val_adr, y = Y_val, model = model_file,
            batch_size = 32, accuracy_operation = accuracy_operation, 
            X_ph = X_ph, Y_ph = Y_ph, keep_p_ph = keep_p_ph)

    print('Epoch {}/{}, Validation accuracy: {:.3f}'.format(epoch+1, EPOCH_NO, accuracy))
    print('Epoch {}/{}, Validation accuracy: {:.3f}'.format(epoch+1, EPOCH_NO, accuracy), 
            file = fout)
