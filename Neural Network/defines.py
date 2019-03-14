from data import *
import os

# Save Path
dir_path = os.path.abspath(__file__+"/../")

# Load Dataset
print("\nSTART: Data preparation.")
augment = True
augment_size = 15000
mnist_data = MNIST()
if augment == True: mnist_data.data_augmentation(augment_size=augment_size)
train_size = mnist_data.train_size
test_size = mnist_data.test_size
print("DONE: ", train_size, test_size, "\n")

# Dataset Variables
classes = 10
width = 28
height = 28
depth = 1
# Training Hyperparameters
keep_prob_conv = 0.6
keep_prob_fc = 0.5
epochs = 10
train_batch_size = 64
learning_rate = 5e-4
# Model Hyperparameters
dense_size = 512
stddev = 0.050
bias_weight_init = 0.050
# Helper variables
test_batch_size = 1000
train_errors, test_errors = [], []
train_accs, test_accs = [], []
train_mini_batches = train_size//train_batch_size + 1
test_mini_batches = test_size//test_batch_size + 1