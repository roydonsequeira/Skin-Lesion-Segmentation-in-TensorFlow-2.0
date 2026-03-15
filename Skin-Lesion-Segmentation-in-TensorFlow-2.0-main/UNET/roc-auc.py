import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Function to calculate and plot ROC-AUC
# Function to calculate and plot ROC-AUC
def plot_roc_auc(y_true, y_scores, classes):
    n_classes = len(classes)

    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=classes)

    # Check if y_scores has only one column
    if y_scores.ndim == 1 or y_scores.shape[1] == 1:
        fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')

    else:
        # Initialize the figure
        plt.figure(figsize=(8, 8))

        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # Set labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        # Show the plot
        plt.show(block=True)

# Example usage
if __name__ == "__main__":
    # Load your data (replace these lines with your data loading logic)
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_scores = np.random.rand(len(y_true))  # Replace with your model's probability scores
    classes = [0, 1]

    # Plot ROC-AUC curve
    plot_roc_auc(y_true, y_scores, classes)
