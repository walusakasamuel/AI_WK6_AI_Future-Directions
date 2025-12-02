```python
# train_model.py
# Train a lightweight image classifier using transfer learning (MobileNetV2)

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths (adjust as needed)
DATA_DIR = 'test_dataset'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
MODEL_SAVE = 'saved_model'

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   rotation_range=20, horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE, subset='training')

val_gen = train_datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE, subset='validation')

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(train_gen.num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model
model.save(MODEL_SAVE)
print('Saved model to', MODEL_SAVE)