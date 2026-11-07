"""Recalculate metrics from the predictions.npz written by train.py."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate(path, output):
    with np.load(path, allow_pickle=False) as data:
        gold, pred, classes = data['gold'], data['predictions'], data['classes']
    if gold.shape != pred.shape or gold.ndim != 2 or gold.shape[1] != len(classes):
        raise ValueError('Gold, prediction, and label dimensions must agree')
    if not np.isin(gold, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('Expected binary label matrices')
    p, r, f, support = precision_recall_fscore_support(gold, pred, average=None, zero_division=0)
    metrics = {
        'Joint Macro-F1': float(f.mean()),
        'Joint Micro-F1': float(precision_recall_fscore_support(gold, pred, average='micro', zero_division=0)[2]),
        'Macro-Precision': float(p.mean()), 'Macro-Recall': float(r.mean()),
        'Joint Accuracy': float(accuracy_score(gold, pred)),
        'exact_matches': int((gold == pred).all(axis=1).sum()), 'samples': len(gold),
    }
    tp = ((gold == 1) & (pred == 1)).sum(axis=0)
    fp = ((gold == 0) & (pred == 1)).sum(axis=0)
    fn = ((gold == 1) & (pred == 0)).sum(axis=0)
    tn = ((gold == 0) & (pred == 0)).sum(axis=0)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(dict(label=classes, TP=tp, FP=fp, FN=fn, TN=tn, precision=p, recall=r, f1=f, support=support)).to_csv(output / 'per_label_metrics.csv', index=False)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output-dir', default='outputs/recalculated')
    args = parser.parse_args()
    evaluate(args.predictions, args.output_dir)
