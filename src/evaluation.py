from sklearn.metrics import roc_auc_score, f1_score, roc_curve

def compute_auc(labels, scores):
    """Compute standard ROC-AUC."""
    return roc_auc_score(labels, scores)

def compute_f1(labels, scores, threshold=0.5):
    """Compute F1 score at a fixed threshold."""
    preds = [1 if s >= threshold else 0 for s in scores]
    return f1_score(labels, preds)

def compute_tpr_at_fpr(labels, scores, fpr_threshold=0.0001):
    """
    Compute TPR at a given FPR threshold.
    fpr_threshold = 0.01% = 0.0001
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find the max TPR where FPR <= threshold
    tpr_at_threshold = tpr[fpr <= fpr_threshold]
    if len(tpr_at_threshold) == 0:
        return 0.0  # no point below threshold
    return tpr_at_threshold[-1]