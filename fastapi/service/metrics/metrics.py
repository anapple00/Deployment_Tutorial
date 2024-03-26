from sklearn.metrics import precision_score, recall_score, f1_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def calculate_metrics(preds, labels):
    acc = simple_accuracy(preds, labels)
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
