import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def calculate_metrics(results):
    """
    Counts the confusion matrix and calculates the accuracy.
    results: list of dicts {'true_id': '001', 'pred_id': '001'}
    """
    tp = 0 # TP
    fp = 0 # FP
    tn = 0 # TN
    fn = 0 # FN
    
    for r in results:
        if r['true_id'] == r['pred_id']:
            if r['true_id'] == "Unknown": tn += 1
            else: tp += 1
        else:
            if r['pred_id'] == "Unknown": fn += 1
            else: fp += 1
            
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0
    return {"accuracy": accuracy, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def plot_roc_curve(genuine_scores, imposter_scores, out_path):
    """Generates an ROC curve from matching scores."""
    y_true = [1] * len(genuine_scores) + [0] * len(imposter_scores)
    y_scores = list(genuine_scores) + list(imposter_scores)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(results, out_path, num_classes=10):
    """Generates a normalized confusion matrix for the first N classes."""
    # Filter results for only the first N identities to keep plot clean
    ids = sorted(list(set([r['true_id'] for r in results if r['true_id'] != "Unknown"])))[:num_classes]
    filtered = [r for r in results if r['true_id'] in ids and r['pred_id'] in ids]
    
    y_true = [r['true_id'] for r in filtered]
    y_pred = [r['pred_id'] for r in filtered]
    
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=ids, yticklabels=ids)
    plt.title(f"Normalized Confusion Matrix (First {num_classes} Identities)")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path)
    plt.close()