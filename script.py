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

X_train_adr, Y_train = parse_txt(training_data)
X_val_adr, Y_val = parse_txt(validation_data)
X_test_adr, Y_test = parse_txt(test_data)

train(X = X_train_adr, y = Y_train, model = model_file,
        batch_size = 32, keep_prob = 0.75, training_operation = training_operation, 
        X_ph = X_ph, Y_ph = Y_ph, keep_p_ph = keep_p_ph)

accuracy = evaluate(X = X_val_adr, y = Y_val, model = model_file,
        batch_size = 32, accuracy_operation = accuracy_operation, 
        X_ph = X_ph, Y_ph = Y_ph, keep_p_ph = keep_p_ph)

print('Validation accuracy: {:.3f}'.format(accuracy))
