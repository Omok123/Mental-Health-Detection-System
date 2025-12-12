# src/evaluate.py
"""
Evaluation module for test set
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.improved_model_v2 import RAF_DB_MultiAttribute_Net

class SimpleImageDataset(Dataset):
    """Simple dataset for evaluation"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # RAF-DB Standard Emotion Order (CORRECTED)
        # Index: 0=surprise, 1=fear, 2=disgust, 3=happy, 4=sad, 5=angry, 6=neutral
        self.emotions = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        self.emotion_to_idx = {em: i for i, em in enumerate(self.emotions)}
        
        # Collect all images
        self.samples = []
        for emotion in self.emotions:
            emotion_folder = self.image_dir / emotion
            if emotion_folder.exists():
                for img_path in emotion_folder.glob('*.jpg'):
                    self.samples.append((str(img_path), self.emotion_to_idx[emotion]))
        
        print(f"Loaded {len(self.samples)} images from {image_dir}")
        
        # Print distribution
        emotion_counts = {}
        for _, label in self.samples:
            emotion = self.emotions[label]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("Dataset distribution:")
        for emotion in self.emotions:
            count = emotion_counts.get(emotion, 0)
            print(f"  {emotion:10s}: {count:4d} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ModelEvaluator:
    """Evaluate model on test set"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # RAF-DB Standard Emotion Order (CORRECTED)
        # Index: 0=surprise, 1=fear, 2=disgust, 3=happy, 4=sad, 5=angry, 6=neutral
        self.emotions = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
    
    def evaluate(self, test_loader, use_correction=True):
        """
        Evaluate model on test set
        
        Args:
            test_loader: DataLoader with test data
            use_correction: Use multi-attribute correction
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\nEvaluating model...")
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if use_correction:
                    corrected_pred, emotion_probs, _, _, _ = self.model.predict_with_correction(images)
                    preds = corrected_pred
                else:
                    emotion_logits, _, _, _ = self.model(images)
                    emotion_probs = F.softmax(emotion_logits, dim=1)
                    preds = torch.argmax(emotion_probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(emotion_probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(all_labels, all_preds, average=None)
        
        results = {
            'accuracy': accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'per_class': {
                'precision': per_class_precision,
                'recall': per_class_recall,
                'f1_score': per_class_f1,
                'support': per_class_support
            }
        }
        
        return results
    
    def plot_confusion_matrix(self, labels, predictions, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.emotions,
            yticklabels=self.emotions,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - RAF-DB Test Set', fontsize=16, pad=20)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def print_classification_report(self, labels, predictions):
        """Print classification report"""
        report = classification_report(
            labels,
            predictions,
            target_names=self.emotions,
            digits=4
        )
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(report)
        print("="*70)
        
        return report
    
    def analyze_confusion_patterns(self, labels, predictions):
        """Analyze specific confusion patterns (e.g., SAD vs HAPPY)"""
        cm = confusion_matrix(labels, predictions)
        
        print("\n" + "="*70)
        print("CONFUSION PATTERN ANALYSIS")
        print("="*70)
        
        # Analyze SAD vs HAPPY (using corrected indices)
        sad_idx = self.emotions.index('sad')      # Index 4
        happy_idx = self.emotions.index('happy')  # Index 3
        
        sad_as_happy = cm[sad_idx][happy_idx]
        happy_as_sad = cm[happy_idx][sad_idx]
        total_sad = cm[sad_idx].sum()
        total_happy = cm[happy_idx].sum()
        
        sad_as_happy_pct = (sad_as_happy / total_sad) * 100 if total_sad > 0 else 0
        happy_as_sad_pct = (happy_as_sad / total_happy) * 100 if total_happy > 0 else 0
        
        print(f"\n📊 SAD vs HAPPY Confusion:")
        print(f"   Sad → Happy:  {sad_as_happy}/{total_sad} ({sad_as_happy_pct:.2f}%)")
        print(f"   Happy → Sad:  {happy_as_sad}/{total_happy} ({happy_as_sad_pct:.2f}%)")
        
        if sad_as_happy_pct < 10 and happy_as_sad_pct < 10:
            print("   ✅ EXCELLENT! Confusion < 10%")
        elif sad_as_happy_pct < 20 and happy_as_sad_pct < 20:
            print("   🟢 GOOD! Confusion < 20%")
        else:
            print("   ⚠️  Needs improvement")
        
        # Top confusions
        print(f"\n📊 Top 5 Confusion Patterns:")
        confusion_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > 0:
                    confusion_pairs.append((
                        self.emotions[i],
                        self.emotions[j],
                        cm[i][j],
                        (cm[i][j] / cm[i].sum()) * 100 if cm[i].sum() > 0 else 0
                    ))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"   {'True':<10} {'Pred':<10} {'Count':<8} {'%':<8}")
        print("   " + "-"*40)
        for true_em, pred_em, count, pct in confusion_pairs[:5]:
            print(f"   {true_em:<10} {pred_em:<10} {count:<8} {pct:>6.2f}%")

def create_test_dataloader(test_dir, batch_size=32):
    """Create test dataloader"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleImageDataset(test_dir, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return loader