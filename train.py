import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset_helper import read_cifar_10
from alexnet import AlexNet


# Model Hyperparameters
INPUT_WIDTH = 70
INPUT_HEIGHT = 70
INPUT_CHANNELS = 3
NUM_CLASSES = 10

LEARNING_RATE = 0.001   # Reduced from 0.01 to stabilize training
MOMENTUM = 0.9
KEEP_PROB = 0.5

EPOCHS = 100
BATCH_SIZE = 128

# Load CIFAR-10 dataset
print('Reading CIFAR-10...')
X_train, Y_train, X_test, Y_test = read_cifar_10(image_width=INPUT_WIDTH, image_height=INPUT_HEIGHT)

# ✅ Apply Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,       # Randomly rotate images by 15 degrees
    width_shift_range=0.1,   # Random horizontal shifts
    height_shift_range=0.1,  # Random vertical shifts
    horizontal_flip=True,    # Flip images horizontally
    zoom_range=0.2,          # Randomly zoom in
    brightness_range=[0.8,1.2]  # Adjust brightness
)

# Fit the generator to training data
datagen.fit(X_train)

# Initialize AlexNet
alexnet = AlexNet(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, input_channels=INPUT_CHANNELS,
                  num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, momentum=MOMENTUM, keep_prob=KEEP_PROB)


# just making sure that i have checkpoints to train back to
# Define checkpoint path
checkpoint_path = "./model/alexnet_checkpoint.weights.h5"
from tensorflow.keras.callbacks import ModelCheckpoint

# Check if checkpoint exists before loading
if os.path.exists(checkpoint_path):
    alexnet.model.load_weights(checkpoint_path)
    print("✅ Loaded saved weights. Resuming training...")
else:
    print("⚠️ No checkpoint found. Training from scratch...")

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath="./model/alexnet_checkpoint.weights.h5",  # Save to this file
    save_weights_only=True,  # Saves only weights, not architecture
    monitor="val_loss",  # Save best model based on validation loss
    save_best_only=True,  # Overwrite only if the model improves
    verbose=1
)


print('Training dataset with Data Augmentation...')

# ✅ Train using the augmented data
# alexnet.train(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),  
#               epochs=EPOCHS, validation_data=(X_test, Y_test))


alexnet.model.fit(
    datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, Y_test),
    #workers=4,  # Use 4 CPU workers for data loading
    # use_multiprocessing=True,  # Enable multiprocessing for efficiency
   #  max_queue_size=10  # Control queue size for multiprocessing
    callbacks=[checkpoint_callback]  # Add checkpoint callback
)



# Evaluate the model
final_train_accuracy = alexnet.evaluate(X_train, Y_train)
final_test_accuracy = alexnet.evaluate(X_test, Y_test)

print(f'Final Train Accuracy = {final_train_accuracy[1]:.3f}')
print(f'Final Test Accuracy = {final_test_accuracy[1]:.3f}')

# Save the model
print('Saving as HDF5...')
alexnet.save('./model/alexnet.weights.h5')

print('Saving as Keras file...')
alexnet.save('./model/alexnet.keras')

print('Model saved.')
print('Training done successfully.')
