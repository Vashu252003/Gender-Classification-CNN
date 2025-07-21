# main.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Data Preparation & Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    "data/Training/",
    batch_size=256,
    class_mode='binary',
    target_size=(64, 64)
)

validation_generator = test_datagen.flow_from_directory(
    "data/Validation/",
    batch_size=256,
    class_mode='binary',
    target_size=(64, 64)
)

# 2. Build the CNN Model
model = Sequential([
    # 1st conv
    tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
    # 2nd conv
    tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    # 3rd conv
    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    # 4th conv
    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    # 5th conv
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
    # Flatten and Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compile and Train
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

hist = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // 256,
    validation_steps=validation_generator.samples // 256,
    epochs=1
)

# Save the trained model
model.save('saved_model/gender_classification_model.h5')

# 4. Evaluate the model
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.savefig('training_validation_accuracy.png')
plt.show()

# 5. Test on a new image
path = "data/Validation/female/112944.jpg.jpg"
img = load_img(path, target_size=(64, 64))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0

classes = model.predict(x, batch_size=1)
print(classes[0])

if classes[0] > 0.5:
    print("is a man")
else:
    print("is a female")
plt.imshow(img)
plt.show()

