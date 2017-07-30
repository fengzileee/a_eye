from simplenet import *
from aeye import *

model_name = 'model_trial.ckpt'
model_dir = './model_trial'
model_file = model_dir + '/' + model_name

training_data = './BreaKHis_data/split1/200X_train.txt'
validation_data = './BreaKHis_data/split1/200X_val.txt'
test_data = './BreaKHis_data/split1/200X_test.txt'


#training_operation, accuracy_operation, predict_operation \
#        = construct_graph(network_builder = network, learning_rate = 0.01)

X_adr, Y = parse_txt(training_data)

print(X_adr[20], Y[20])
