In the name of God

# AlexNet
This repository contains implementation of [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
(ImageNet Classification with Deep Convolutional Neural Networks) by Tensorflow and the network tested with the
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

![AlexNet Architecture](alexnet.png)

# Download the CIFAR-10 dataset
Before train and evaluate the network, you should download the following dataset:

* CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Extract the `cifar-10-python.tar.gz` file, then your folder structure should be like the following image:

![Folder Structure](folder_structure.png)

# Downloading the pretrained model

https://www.kaggle.com/models/kwadwosarbengbaafi/alexnet-model-using-the-cifar10-dataset

# Training CIFAR-10 dataset
To train the network with cifar-10 dataset, type the following command at the command prompt:
```
python3 ./train.py
```

Sample images from cifar-10 dataset:

![cifar_10_sample](cifar_10_sample.jpg)

## Results


### Epoch 1
```
Train Accuracy = 0.1718
Validation Accuracy = 0.2737
```

### Epoch 2
```
Train Accuracy = 0.4015
Validation Accuracy =  0.5040
```

...

### Epoch 50
```
Train Accuracy = 0.8430
Validation Accuracy = 0.8105
```

...

### Epoch 100
```
Final Train Accuracy = 0.957
Final Validation Accuracy = 0.827
```

# Evaluating CIFAR-10 dataset
To evaluate the network with cifar-10 dataset, type the following command at the command prompt:
```
python3 ./evaluate.py
```
# Running a Test with the Pretrained Model
```
Open testing_cifar10.ipynb

Run the code within

```

# Dependencies
* Python 3
* numpy
* scipy
* pillow
* tensorflow

# Links
* https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
* https://www.cs.toronto.edu/~kriz/cifar.html
* https://github.com/Sarbeng/AlexNet_model
