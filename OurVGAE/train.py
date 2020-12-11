import random
import time
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from optimizer import OptimizerAE, OptimizerVAE
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Train on CPU (hide GPU) due to memory constraints

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
logs_path = '/tmp/tensorflow_logs/example/'



# for reproducibility
random.seed(0)
np.random.seed(0)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')

# ARASH ADDED: changing gcn_ae to gcn_vae
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load Data

#################################################################################################################
# ARASH ADDED: changing the dataset to ours since we have the adj and the features
def load_our_dataset():
    # making our feature sparse values:
    features = pd.read_csv("Oct2019_10min_450_Jan_Feb_V.csv")
    features_true = features.values
    # removing 80 percent of features in each column randomly
    # ind_to_zero = random.sample(range(features.shape[1]), int((features.shape[1] * 0.8)))
    features = features.applymap(lambda x: x if random.uniform(0, 1) < 0.50 else float(0)).values
    # turning adj matrix to 0 and 1
    adj = pd.read_csv("Oct2019_10min_450_Jan_Feb_W.csv", header=None)
    adj = adj.applymap(lambda x: float(1) if x > 0 else x)
    adj = adj.values

    return adj, features.T, features_true.T

def load_our_dataset_TTI():
    # making our feature sparse values:
    features = pd.read_csv("jan_speed.csv", header=None)
    features_true = features.values
    # removing 80 percent of features in each column randomly
    # ind_to_zero = random.sample(range(features.shape[1]), int((features.shape[1] * 0.8)))
    features = features.applymap(lambda x: x if random.uniform(0, 1) < 0.9 else float(0)).values
    # turning adj matrix to 0 and 1
    adj = pd.read_csv("adj.csv", header=None)
    # adj = adj.applymap(lambda x: float(1) if x > 0 else x)
    adj = adj.values

    return adj, features, features_true


adj, features, features_true = load_our_dataset()


features_training = features_true


features_training_non_zero_index = np.ones(features.shape) * (features != 0)
count_training = features_training_non_zero_index.sum()


features_training_zero_index = 1 - features_training_non_zero_index
count_test = features_training_zero_index.sum()

# Scipy matrix
adj = sp.csr_matrix(adj)
features = sp.lil_matrix(features_training)
adj_train = adj

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}
# tf.placeholder
num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                          pos_weight=pos_weight,
                          norm=norm))
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.sparse_tensor_to_dense(placeholders['features']),
                           model=model,
                           num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           index_non=tf.constant(features_training_non_zero_index, dtype='float32'),
                           count_non=count_training,
                           index_zero=tf.constant(features_training_zero_index, dtype='float32'),
                           count_zero=count_test
                           )

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


adj_label = sp.csr_matrix(features_training)

adj_label = sparse_to_tuple(adj_label)

toPlot = []

hold = []

# summary = tf.summary.scalar(name="cost", tensor=opt.cost)
# writer = tf.summary.FileWriter('./graphs', sess.graph)

# merged = tf.summary.merge_all()
# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.train_mae, opt.test_mae, opt.train_mse, opt.train_mape, opt.test_mape, opt.TFMAPE], feed_dict=feed_dict)
    outs_2 = sess.run([opt.cost, opt.kl, opt.masked_true, opt.accuracy, opt.masked_training, opt.masked_true_test, opt.masked_training_test, opt.preds_sub, opt.labels_sub], feed_dict=feed_dict)
    hold = sess.run([opt.preds_sub, opt.labels_sub], feed_dict=feed_dict)
    outs_3 = sess.run([model.z_log_std, model.z_mean], feed_dict=feed_dict)

    # print(outs_3)

    # Compute average loss
    avg_cost = outs[1]
    # avg_accuracy = outs[2]
    toPlot.append([outs[4], outs_2[3]])
    # roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    # val_roc_score.append(roc_curr)
    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(avg_cost),
          "kl= ", "{:.5f}".format(outs_2[1]),
          "train_mse=", "{:.5f}".format(outs[4]),
          "test_mse=", "{:.5f}".format(outs_2[3]),
          "train_mae=", "{:.5f}".format(outs[2]),
          "test_mae=", "{:.5f}".format(outs[3]),
          "time=", "{:.5f}".format(time.time() - t))


# print(count_test, count_training)

print("Optimization Finished!")


# np.savetxt("70_30_Pred.csv", hold[0], delimiter=",")
# np.savetxt("70_30_Actual.csv", hold[1], delimiter=",")

toPlot = np.array(toPlot)

plt.plot(range(toPlot.shape[0]), toPlot[:, 0])
plt.plot(range(toPlot.shape[0]), toPlot[:, 1])

plt.title("MSE Train (Blue) Vs. Test (Orange) Per Epoch")
plt.xlabel("MSE")
plt.ylabel("Epoch")


plt.show()


from sklearn.decomposition import NMF

adj, features, features_true = load_our_dataset()


features_training = np.abs(features_true)


features_training_non_zero_index = np.ones(features.shape) * (features != 0)
count_training = features_training_non_zero_index.sum()


features_training_zero_index = 1 - features_training_non_zero_index
count_test = features_training_zero_index.sum()

features_training_train = features_training_non_zero_index * features_training
features_training_test = features_training_zero_index * features_training

model = NMF(n_components=5, init='random', random_state=0, beta_loss='kullback-leibler', solver='mu')
W_train = model.fit_transform(features_training_train)
H_train = model.components_

appr_train = np.matmul(W_train, H_train)
appr_train = features_training_non_zero_index * appr_train

from sklearn.metrics import mean_squared_error
from math import sqrt

mse_train = (1/count_training) * ((features_training_train - appr_train) ** 2).sum()

mae_train = (1/count_training) * np.abs(features_training_train - appr_train).sum()
print('train mae: ', mae_train)
print('train mse: ', mse_train)

W_test = model.transform(features_training_test)
H_test = model.components_

appr_test = np.matmul(W_test, H_test)
appr_test = features_training_zero_index * appr_test

mse_test = (1/count_test) * ((features_training_test - appr_test) ** 2).sum()
mae_test = (1/count_test) * np.abs(features_training_test - appr_test).sum()
print('test mape: ', mae_test)
print('test mse: ', mse_test)