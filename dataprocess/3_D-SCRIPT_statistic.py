import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,precision_recall_curve,classification_report, auc, mean_squared_error

def classification_metrics(targets, predictions, threshold=0.5):
    binary_predictions = (predictions >= threshold).astype(int)
    accuracy = accuracy_score(targets, binary_predictions)
    f1 = f1_score(targets, binary_predictions)
    precision = precision_score(targets, binary_predictions, pos_label=1)
    recall = recall_score(targets, binary_predictions, pos_label=1)
    auc_score = roc_auc_score(targets, predictions)
    precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions)
    auprc = auc(recall_vals, precision_vals)

    return {
        "Accuracy": accuracy,
        "AUPRC": auprc,
        "F1 Score": f1,
        "Precise":precision,
        "Recall":recall,
        "AUROC": auc_score,
        'y_probs': predictions,
        'y_pred': binary_predictions,
        'y_true': targets
    }

import pandas as pd

df = pd.read_table(sys.argv[1], names=['col1', 'col2', 'predictions','targets'])
aa = classification_metrics(df['targets'], df['predictions'], threshold=0.5)

print(aa)
