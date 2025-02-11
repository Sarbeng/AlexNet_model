import pickle
import numpy as np
import cv2


def __unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar_10(image_width, image_height):
    batches = [f'./cifar-10/data_batch_{i}' for i in range(1, 6)]
    test_batch = './cifar-10/test_batch'

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    train_data = [__unpickle(batch) for batch in batches]
    test_data = __unpickle(test_batch)

    total_train_samples = sum(len(batch[b'labels']) for batch in train_data)
    total_test_samples = len(test_data[b'labels'])

    X_train = np.zeros((total_train_samples, image_width, image_height, 3), dtype=np.uint8)
    Y_train = np.zeros((total_train_samples, len(classes)), dtype=np.float32)

    index = 0
    for batch in train_data:
        for i in range(len(batch[b'labels'])):
            image = batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
            label = batch[b'labels'][i]

            X = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
            Y = np.zeros(len(classes), dtype=np.int32)
            Y[label] = 1

            X_train[index + i] = X
            Y_train[index + i] = Y
        
        index += len(batch[b'labels'])

    X_test = np.zeros((total_test_samples, image_width, image_height, 3), dtype=np.uint8)
    Y_test = np.zeros((total_test_samples, len(classes)), dtype=np.float32)

    for i in range(len(test_data[b'labels'])):
        image = test_data[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
        label = test_data[b'labels'][i]

        X = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        Y = np.zeros(len(classes), dtype=np.int32)
        Y[label] = 1

        X_test[i] = X
        Y_test[i] = Y

    return X_train, Y_train, X_test, Y_test
