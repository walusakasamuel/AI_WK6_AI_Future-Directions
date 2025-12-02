#!/usr/bin/env python3
"""
Edge AI Prototype: Train a lightweight image classification model for recyclable items
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.model_selection import train_test_split

def create_dataset():
    """
    Create a synthetic dataset for recyclable items since we don't have a real one.
    In practice, you would use a real dataset like TrashNet or Waste Classification dataset.
    """
    # For demonstration, we'll use CIFAR-10 and map some classes to recyclable items
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    
    # Map CIFAR-10 classes to recyclable item categories
    # Class mapping: 0: paper, 1: plastic, 2: glass, 3: metal
    class_mapping = {
        0: 0,  # airplane -> paper
        1: 1,  # automobile -> plastic
        2: 2,  # bird -> glass
        3: 3,  # cat -> metal
        4: 0,  # deer -> paper
        5: 1,  # dog -> plastic
        6: 2,  # frog -> glass
        7: 3,  # horse -> metal
        8: 0,  # ship -> paper
        9: 1   # truck -> plastic
    }
    
    # Filter and map labels
    train_labels_mapped = np.array([class_mapping[label[0]] for label in train_labels])
    test_labels_mapped = np.array([class_mapping[label[0]] for label in test_labels])
    
    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # One-hot encode labels
    train_labels_onehot = keras.utils.to_categorical(train_labels_mapped, 4)
    test_labels_onehot = keras.utils.to_categorical(test_labels_mapped, 4)
    
    return (train_images, train_labels_onehot), (test_images, test_labels_onehot)

def create_lightweight_model(input_shape=(32, 32, 3), num_classes=4):
    """
    Create a lightweight CNN model for edge deployment
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_mobilenet_model(input_shape=(32, 32, 3), num_classes=4):
    """
    Create a MobileNet-based model for better accuracy (still lightweight)
    """
    # Create base model
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,  # We'll train from scratch for demonstration
        alpha=0.35  # Width multiplier for lightweight version
    )
    
    # Freeze base model layers
    base_model.trainable = True
    
    # Create new model on top
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model

def train_model(model_type='lightweight', epochs=20, batch_size=32):
    """
    Train the model and save it
    """
    print("Loading and preparing dataset...")
    (train_images, train_labels), (test_images, test_labels) = create_dataset()
    
    # Split training data for validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Create model
    if model_type == 'mobilenet':
        print("Creating MobileNet-based model...")
        model = create_mobilenet_model()
    else:
        print("Creating lightweight CNN model...")
        model = create_lightweight_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_images, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the model
    model.save('recyclable_classifier.h5')
    print("Model saved as 'recyclable_classifier.h5'")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history, test_accuracy

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a lightweight image classification model')
    parser.add_argument('--model', type=str, default='lightweight',
                       choices=['lightweight', 'mobilenet'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Edge AI Prototype: Recyclable Item Classification")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Train the model
    model, history, test_accuracy = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("=" * 60)