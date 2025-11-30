import tensorflow as tf 
from keras import layers, models

(train_images, train_labels), ( test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0

# Build the CNN model
model_cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# Compile and train the model
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

loss, accuracy = model_cnn.evaluate(test_images, test_labels)
print(f'Test Accuracy: {accuracy}')