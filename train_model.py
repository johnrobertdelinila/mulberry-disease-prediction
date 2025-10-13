#!/usr/bin/env python3
"""
Mulberry Disease Prediction - Training Script
This script trains a CNN model to classify mulberry leaf diseases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
import os
from os import listdir
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, array_to_img
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🌿 Mulberry Disease Prediction - Model Training")
    print("=" * 50)
    
    # Set the dataset path
    path = "Dataset/Mulberry_Data"
    
    # Check if dataset exists
    if not os.path.exists(path):
        print(f"❌ Error: Dataset path '{path}' not found!")
        print("Please make sure the Dataset folder is in the current directory.")
        return
    
    print(f"✅ Dataset found at: {path}")
    
    # Function to count images in each folder
    def count_images_in_folders(directory):
        folder_image_count = {}
        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                image_count = len([f for f in os.listdir(folder_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                folder_image_count[folder_name] = image_count
        return folder_image_count

    # Get the count of images in each folder
    print("\n📊 Dataset Statistics:")
    image_counts = count_images_in_folders(path)
    label_counts = pd.DataFrame(list(image_counts.items()), columns=['Class', 'Count'])
    print(label_counts)
    
    # Function to convert image to array
    def convert_image_to_array(image_dir):
        try:
            image = cv2.imread(image_dir)
            if image is not None:
                image = cv2.resize(image, (256, 256))  # Resizing to 256x256 pixels
                return img_to_array(image)
            else:
                return np.array([])
        except Exception as e:
            print(f"Error processing {image_dir}: {e}")
            return None

    # Lists to hold the image arrays and labels
    print("\n🔄 Loading and preprocessing images...")
    image_list, label_list = [], []

    # Define your categories and their corresponding labels
    all_labels = ['Healthy_Leaves', 'Rust_leaves', 'Spot_leaves', 'deformed_leaves', 'Yellow_leaves']
    binary_labels = [0, 1, 2, 3, 4]  # Your label mapping

    # Convert images to arrays and assign labels
    for directory in all_labels:
        folder_path = os.path.join(path, directory)
        if os.path.exists(folder_path):
            plant_image_list = listdir(folder_path)
            print(f"Processing {directory}: {len(plant_image_list)} images")
            
            for files in plant_image_list:
                image_path = os.path.join(folder_path, files)
                img_array = convert_image_to_array(image_path)
                if img_array is not None:
                    image_list.append(img_array)
                    label_list.append(binary_labels[all_labels.index(directory)])

    print(f"✅ Successfully loaded {len(image_list)} images")

    # Convert lists to NumPy arrays
    image_list = np.array(image_list)
    label_list = np.array(label_list)

    # Normalize the images
    image_list = image_list.astype('float32') / 255.0
    
    # Split the data
    print("\n🔄 Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=10)
    
    # Reshape
    x_train = x_train.reshape(-1, 256, 256, 3)
    x_test = x_test.reshape(-1, 256, 256, 3)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print(f"Training set: {x_train.shape}")
    print(f"Test set: {x_test.shape}")

    # Model architecture
    print("\n🏗️ Building CNN model...")
    model = Sequential()

    # First Convolutional Layer
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(256, 256, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Layer
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Layer
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(all_labels), activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    
    print("✅ Model compiled successfully!")
    print(f"Model summary:")
    model.summary()

    # Splitting the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)

    # Training the model
    print("\n🚀 Starting training...")
    epochs = 50
    batch_size = 64
    
    history = model.fit(
        x_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_data=(x_val, y_val),
        verbose=1
    )

    # Plotting accuracy and loss
    print("\n📈 Plotting training results...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], color='r', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], color='b', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], color='r', label='Training Loss')
    plt.plot(history.history['val_loss'], color='b', label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    # Evaluating model performance
    print("\n📊 Evaluating model performance...")
    scores = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {scores[1] * 100:.2f}%")
    print(f"Test Loss: {scores[0]:.4f}")

    # Make predictions
    print("\n🔮 Making predictions on test set...")
    y_pred = model.predict(x_test)
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(10, len(x_test))):
        true_label = all_labels[np.argmax(y_test[i])]
        pred_label = all_labels[np.argmax(y_pred[i])]
        confidence = np.max(y_pred[i]) * 100
        print(f"True: {true_label:<15} | Predicted: {pred_label:<15} | Confidence: {confidence:.1f}%")

    # Saving the model
    model_path = "Model/mulberry_leaf_disease_model_enhanced.h5"
    os.makedirs("Model", exist_ok=True)
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("\n🎉 Training completed successfully!")
    print("You can now use this model for predictions.")

if __name__ == "__main__":
    main()
