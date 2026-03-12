import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm
import json


def evaluate_model(model, dataloader, device, is_multitask=False):
    """
    Evaluate a trained model.
    Returns overall metrics, per-manipulation accuracy, and confusion matrix.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    manip_preds = defaultdict(list)   # manip_type -> list of (pred, label)

    with torch.no_grad():
        for images, labels, manip_types in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)

            if is_multitask:
                binary_out, _ = model(images)
                outputs = binary_out
            else:
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            all_probs.extend(probs_np)

            for pred, label, manip in zip(preds_np, labels_np, manip_types):
                manip_preds[manip].append((int(pred), int(label)))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall metrics
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        'accuracy':  float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall':    float(recall_score(all_labels, all_preds, zero_division=0)),
        'f1':        float(f1_score(all_labels, all_preds, zero_division=0)),
        'auc':       float(roc_auc_score(all_labels, all_probs)),
        'confusion_matrix': cm.tolist(),
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1]),
        'false_negative_rate': float(cm[1, 0] / (cm[1, 0] + cm[1, 1])) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0,
    }

    # Per-manipulation accuracy
    per_manip = {}
    for manip, pairs in sorted(manip_preds.items()):
        preds_m = np.array([p for p, _ in pairs])
        labels_m = np.array([l for _, l in pairs])
        per_manip[manip] = {
            'accuracy': float(accuracy_score(labels_m, preds_m)),
            'n_samples': len(pairs),
        }
    metrics['per_manipulation'] = per_manip

    _print_results(metrics, cm)
    return metrics


def _print_results(metrics, cm):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:            {metrics['accuracy']*100:.2f}%")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1-Score:            {metrics['f1']:.4f}")
    print(f"AUC-ROC:             {metrics['auc']:.4f}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.4f}  "
          f"(fake predicted as real)")

    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Real    Fake")
    print(f"Actual  Real    {cm[0,0]:<7} {cm[0,1]:<6}")
    print(f"        Fake    {cm[1,0]:<7} {cm[1,1]:<6}")

    print("\nPer-Manipulation Accuracy:")
    for manip, stats in metrics['per_manipulation'].items():
        print(f"  {manip:<20} {stats['accuracy']*100:.2f}%  (n={stats['n_samples']})")
    print("=" * 60 + "\n")


def save_metrics(metrics, save_path):
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {save_path}")
