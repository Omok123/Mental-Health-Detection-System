# test_happy_class.py
from src.evaluate import ModelEvaluator, create_test_dataloader
from src.improved_model_v2 import RAF_DB_MultiAttribute_Net
import torch

# Load model
model, _ = RAF_DB_MultiAttribute_Net.load_from_checkpoint(
    'checkpoints/rafdb_multi_attribute_final.pth',
    device='cpu'
)

# Create test loader
test_loader = create_test_dataloader('data/test_organized', batch_size=32)

# Evaluate
evaluator = ModelEvaluator(model, device='cpu')
results = evaluator.evaluate(test_loader, use_correction=True)

# Check happy-specific performance
from sklearn.metrics import classification_report
emotions = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

print("\n" + "="*70)
print("HAPPY CLASS PERFORMANCE")
print("="*70)

report = classification_report(
    results['labels'],
    results['predictions'],
    target_names=emotions,
    digits=4
)

print(report)

# Check what happy is being predicted as
import numpy as np
happy_idx = 3
happy_mask = results['labels'] == happy_idx
happy_predictions = results['predictions'][happy_mask]

from collections import Counter
prediction_counts = Counter(happy_predictions)

print("\nWhat are HAPPY faces being predicted as?")
for pred_idx, count in prediction_counts.most_common():
    percentage = (count / len(happy_predictions)) * 100
    print(f"  {emotions[pred_idx]:10s}: {count:4d} ({percentage:5.2f}%)")