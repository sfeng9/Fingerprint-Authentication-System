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