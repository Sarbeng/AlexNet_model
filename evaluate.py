# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, models
from alexnet import AlexNet
from dataset_helper import read_cifar_10

INPUT_WIDTH = 70
INPUT_HEIGHT = 70
INPUT_CHANNELS = 3

NUM_CLASSES = 10

LEARNING_RATE = 0.001   # Original value: 0.01
MOMENTUM = 0.9
KEEP_PROB = 0.5

EPOCHS = 100
BATCH_SIZE = 128

print('Reading CIFAR-10...')
X_train, Y_train, X_test, Y_test = read_cifar_10(image_width=INPUT_WIDTH, image_height=INPUT_HEIGHT)

# # Convert the input data to float32 and normalize to the range [0, 1]
# X_train = X_train.astype('float32') / 255.0
# X_test = X_test.astype('float32') / 255.0


# Create the AlexNet model
alexnet = AlexNet(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, input_channels=INPUT_CHANNELS,
                  num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, momentum=MOMENTUM, keep_prob=KEEP_PROB)

# Compile the model
alexnet.compile_model()

# Load the pre-trained weights
print('Loading weights...')
alexnet.load('./model/alexnet.keras')

# Evaluate dataset
print('Evaluating dataset...')
train_loss, train_accuracy = alexnet.evaluate(X_train, Y_train)
test_loss, test_accuracy = alexnet.evaluate(X_test, Y_test)

print('Train Accuracy = {:.3f}'.format(train_accuracy))
print('Test Accuracy = {:.3f}'.format(test_accuracy))
