"""
visualization_utils.py

Helper functions for plotting confusion matrices, metrics, or quantum circuits.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(labels, predictions, class_names=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix given true labels and predictions.
    """
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()
