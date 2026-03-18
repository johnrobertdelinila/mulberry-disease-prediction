"""
Generate Confusion Matrix using ONLY the 20% unseen test images.

This script replicates the exact same train/test split (random_state=10)
used during training, then evaluates the model ONLY on the 20% test set
that the model has NEVER seen during training.

Usage:
    python generate_test_confusion_matrix.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from os import listdir
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)


# === CONFIGURATION ===
DATASET_PATH = "Dataset/Mulberry_Data"
MODEL_PATH = "Model/mulberry_leaf_disease_model_enhanced5.h5"
CLASS_LABELS = ['Healthy_Leaves', 'Rust_leaves', 'Spot_leaves', 'deformed_leaves', 'Yellow_leaves']
IMAGE_SIZE = (256, 256)
TEST_SIZE = 0.2
RANDOM_STATE = 10  # Same as training to get the exact same split


def convert_image_to_array(image_dir):
    """Convert image to array (same as training script)."""
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, IMAGE_SIZE)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error processing {image_dir}: {e}")
        return None


def load_dataset():
    """Load dataset (same process as training)."""
    image_list, label_list = [], []
    binary_labels = list(range(len(CLASS_LABELS)))

    for directory in CLASS_LABELS:
        folder_path = os.path.join(DATASET_PATH, directory)
        if os.path.exists(folder_path):
            plant_image_list = listdir(folder_path)
            print(f"Loading {directory}: {len(plant_image_list)} images")

            for files in plant_image_list:
                image_path = os.path.join(folder_path, files)
                img_array = convert_image_to_array(image_path)
                if img_array is not None:
                    image_list.append(img_array)
                    label_list.append(binary_labels[CLASS_LABELS.index(directory)])

    return np.array(image_list), np.array(label_list)


def plot_confusion_matrix(cm, accuracy, output_path="confusion_matrix_20pct_test.png"):
    """Generate a professional confusion matrix figure."""
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        linewidths=1, linecolor='white',
        annot_kws={"size": 14, "fontweight": "bold"}
    )

    plt.title(
        f'Confusion Matrix — 20% Unseen Test Data\nAccuracy: {accuracy:.2f}%',
        fontsize=14, fontweight='bold', pad=15
    )
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nConfusion matrix saved to: {output_path}")


def plot_normalized_confusion_matrix(cm, accuracy, output_path="confusion_matrix_20pct_test_normalized.png"):
    """Generate a normalized (percentage) confusion matrix figure."""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_normalized, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        linewidths=1, linecolor='white',
        annot_kws={"size": 13, "fontweight": "bold"},
        vmin=0, vmax=100
    )

    plt.title(
        f'Normalized Confusion Matrix — 20% Unseen Test Data\nAccuracy: {accuracy:.2f}%',
        fontsize=14, fontweight='bold', pad=15
    )
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Normalized confusion matrix saved to: {output_path}")


def main():
    print("=" * 60)
    print("Confusion Matrix — 20% Unseen Test Images")
    print("=" * 60)

    # Check paths
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at '{DATASET_PATH}'")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at '{MODEL_PATH}'")
        return

    # Step 1: Load dataset (same as training)
    print("\n[1/5] Loading dataset...")
    image_list, label_list = load_dataset()
    print(f"Total images loaded: {len(image_list)}")

    # Step 2: Normalize (same as training)
    print("\n[2/5] Normalizing images...")
    image_list = image_list.astype('float32') / 255.0

    # Step 3: Split using SAME parameters as training
    print("\n[3/5] Splitting data (test_size=0.2, random_state=10)...")
    x_train, x_test, y_train, y_test = train_test_split(
        image_list, label_list,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"Training set: {x_train.shape[0]} images (used for training)")
    print(f"Test set:     {x_test.shape[0]} images (UNSEEN by model)")

    # Show test set distribution
    unique, counts = np.unique(y_test, return_counts=True)
    print("\nTest set distribution:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {CLASS_LABELS[cls_idx]}: {count} images")

    # Step 4: Load model and predict on TEST SET ONLY
    print("\n[4/5] Loading model and predicting on unseen test data...")
    model = load_model(MODEL_PATH)

    x_test = x_test.reshape(-1, 256, 256, 3)
    y_pred_probs = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Step 5: Generate metrics and confusion matrix
    print("\n[5/5] Generating evaluation metrics...")

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION — 20% UNSEEN TEST DATA ({x_test.shape[0]} images)")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    print(f"{'='*60}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

    # Generate confusion matrices
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, accuracy)
    plot_normalized_confusion_matrix(cm, accuracy)

    print("\nDone! Check the generated PNG files.")


if __name__ == "__main__":
    main()
