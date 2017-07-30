from simplenet import *
from aeye import *

model_name = 'model_trial.ckpt'
model_dir = './model_trial'
model_file = model_dir + '/' + model_name

training_operation, accuracy_operation, predict_operation \
        = construct_graph(network_builder = network, learning_rate = 0.01)

training_data = './BreaKHis_data/split1/400X_train.txt'
validation_data = './BreaKHis_data/split1/400X_val.txt'
test_data = './BreaKHis_data/split1/400X_test.txt'

X_train_adr, Y_train = parse_txt(training_data)

