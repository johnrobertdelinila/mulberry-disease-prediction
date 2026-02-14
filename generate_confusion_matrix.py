#!/usr/bin/env python3
"""
Mulberry Disease Prediction - Confusion Matrix Generator
Generates a confusion matrix from a trained model and dataset.

Usage:
    python generate_confusion_matrix.py
    python generate_confusion_matrix.py --dataset path/to/data --model path/to/model.h5
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")

# Defaults matching train_model.py
DEFAULT_DATASET_PATH = "Dataset/Mulberry_Data"
DEFAULT_MODEL_PATH = "Model/mulberry_leaf_disease_model_enhanced.h5"
ALL_LABELS = [
    "Healthy_Leaves",
    "Rust_leaves",
    "Spot_leaves",
    "deformed_leaves",
    "Yellow_leaves",
]
IMAGE_SIZE = (256, 256)
TEST_SIZE = 0.2
RANDOM_STATE = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a confusion matrix for the Mulberry Disease Prediction model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the dataset directory (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="confusion_matrix.png",
        help="Output filename for the confusion matrix plot (default: confusion_matrix.png)",
    )
    parser.add_argument(
        "--use-all",
        action="store_true",
        help="Evaluate on the entire dataset instead of just the test split",
    )
    return parser.parse_args()


def load_dataset(dataset_path):
    """Load images and labels from the dataset directory."""
    images = []
    labels = []

    for label_idx, label_name in enumerate(ALL_LABELS):
        folder_path = os.path.join(dataset_path, label_name)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: Folder not found: {folder_path}")
            continue

        file_list = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"  {label_name}: {len(file_list)} images")

        for filename in file_list:
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            img = img_to_array(img)
            images.append(img)
            labels.append(label_idx)

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels


def plot_confusion_matrix(cm, output_path):
    """Plot and save a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ALL_LABELS,
        yticklabels=ALL_LABELS,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=12)
    ax.set_ylabel("True Label", fontsize=13, labelpad=12)
    ax.set_title("Mulberry Leaf Disease - Confusion Matrix", fontsize=15, pad=16)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nConfusion matrix saved to: {output_path}")
    plt.show()


def plot_normalized_confusion_matrix(cm, output_path):
    """Plot a percentage-normalized confusion matrix alongside the raw one."""
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    base, ext = os.path.splitext(output_path)
    norm_path = f"{base}_normalized{ext}"

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=ALL_LABELS,
        yticklabels=ALL_LABELS,
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=100,
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=12)
    ax.set_ylabel("True Label", fontsize=13, labelpad=12)
    ax.set_title(
        "Mulberry Leaf Disease - Normalized Confusion Matrix (%)",
        fontsize=15,
        pad=16,
    )
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(norm_path, dpi=150)
    print(f"Normalized confusion matrix saved to: {norm_path}")
    plt.show()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Mulberry Leaf Disease - Confusion Matrix Generator")
    print("=" * 60)

    # --- Validate paths ---
    if not os.path.isdir(args.dataset):
        print(f"\nERROR: Dataset directory not found: {args.dataset}")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"\nERROR: Model file not found: {args.model}")
        print("Train the model first with: python train_model.py")
        sys.exit(1)

    # --- Load model ---
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model)
    print("Model loaded successfully.")

    # --- Load dataset ---
    print(f"\nLoading dataset from: {args.dataset}")
    images, labels = load_dataset(args.dataset)
    print(f"Total images loaded: {len(images)}")

    if len(images) == 0:
        print("\nERROR: No images were loaded. Check your dataset path.")
        sys.exit(1)

    # --- Split or use all data ---
    if args.use_all:
        print("\nEvaluating on the ENTIRE dataset.")
        x_eval, y_eval = images, labels
    else:
        print(
            f"\nSplitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE}) "
            "to match train_model.py..."
        )
        _, x_eval, _, y_eval = train_test_split(
            images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Test set size: {len(x_eval)} images")

    # --- Predict ---
    print("\nRunning predictions...")
    predictions = model.predict(x_eval, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # --- Metrics ---
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, average="weighted")
    recall = recall_score(y_eval, y_pred, average="weighted")
    f1 = f1_score(y_eval, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")

    # --- Classification report ---
    print("\n" + "-" * 60)
    print("  CLASSIFICATION REPORT")
    print("-" * 60)
    report = classification_report(y_eval, y_pred, target_names=ALL_LABELS)
    print(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_eval, y_pred)
    print("-" * 60)
    print("  CONFUSION MATRIX (raw counts)")
    print("-" * 60)
    print(cm)

    # --- Plot ---
    plot_confusion_matrix(cm, args.output)
    plot_normalized_confusion_matrix(cm, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
