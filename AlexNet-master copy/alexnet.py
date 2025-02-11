import tensorflow as tf
from tensorflow.keras import layers, models

class AlexNet:
    def __init__(self, input_width=227, input_height=227, input_channels=3, num_classes=1000, learning_rate=0.01,
                 momentum=0.9, keep_prob=0.5):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.keep_prob = keep_prob
        
        self.model = self.build_model()
        self.compile_model()
    
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', 
                                input_shape=(self.input_height, self.input_width, self.input_channels)))
        model.add(layers.BatchNormalization())  # Replacing Lambda with BatchNormalization
        model.add(layers.MaxPooling2D((3, 3), strides=2))
        
        model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())  # Replacing Lambda with BatchNormalization
        model.add(layers.MaxPooling2D((3, 3), strides=2))
        
        model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((3, 3), strides=2))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(1 - self.keep_prob))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(1 - self.keep_prob))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def compile_model(self):
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    def train(self, train_data, batch_size=None, epochs=100, validation_data=None):
        if isinstance(train_data, tf.keras.preprocessing.image.Iterator):
            self.model.fit(train_data, epochs=epochs, validation_data=validation_data, steps_per_epoch=len(train_data))
        else:
            self.model.fit(train_data[0], train_data[1], batch_size=batch_size, epochs=epochs, validation_data=validation_data)
    
    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, file_path):
        self.model.save(file_path)
    
    def load(self, file_path):
        self.model = tf.keras.models.load_model(file_path, safe_mode=False)
