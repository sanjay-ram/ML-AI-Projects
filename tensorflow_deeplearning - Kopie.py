import tensorflow as tf
from tensorflow.keras import layers, models


(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_activityels, epochs=10, batch_size=32, validation_data=(test_images, test_activityels))

test_loss, test_acc = model.evaluate(test_images, test_activityels)
print(f'Test accuracy: {test_acc}')