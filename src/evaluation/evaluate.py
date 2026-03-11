import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm
import json

def evaluate_model(model, dataloader, device):
    """
    Comprehensive evaluation of a trained model
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("📊 Evaluating model...")
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Evaluation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds)),
        'recall': float(recall_score(all_labels, all_preds)),
        'f1': float(f1_score(all_labels, all_preds)),
        'auc': float(roc_auc_score(all_labels, all_probs))
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    # Print results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {cm[0,0]:<6}  {cm[0,1]:<6}")
    print(f"       Fake   {cm[1,0]:<6}  {cm[1,1]:<6}")
    print("="*60 + "\n")
    
    return metrics

def save_metrics(metrics, save_path):
    """Save metrics to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {save_path}")