# analyze_rafdb_dataset.py
"""
Analyze RAF-DB dataset distribution and class imbalance
"""
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# RAF-DB emotion labels (in standard RAF-DB order)
EMOTIONS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# Directories
TRAIN_DIR = 'data/train_organized'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test_organized'

def count_images_per_emotion(data_dir):
    """Count images for each emotion"""
    counts = {}
    
    if not os.path.exists(data_dir):
        print(f"⚠️  Directory not found: {data_dir}")
        return counts
    
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.exists(emotion_dir):
            images = [f for f in os.listdir(emotion_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            counts[emotion] = len(images)
        else:
            counts[emotion] = 0
    
    return counts

def calculate_imbalance_metrics(counts):
    """Calculate imbalance ratio and statistics"""
    values = [c for c in counts.values() if c > 0]
    
    if not values:
        return None
    
    max_count = max(values)
    min_count = min(values)
    mean_count = np.mean(values)
    std_count = np.std(values)
    
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return {
        'max': max_count,
        'min': min_count,
        'mean': mean_count,
        'std': std_count,
        'imbalance_ratio': imbalance_ratio
    }

def print_distribution(split_name, counts):
    """Print formatted distribution table"""
    total = sum(counts.values())
    
    if total == 0:
        print(f"⚠️  No images found")
        return
    
    print(f"\n{'Emotion':<12} {'Count':<10} {'Percentage':<12} {'Bar':<30}")
    print("-"*70)
    
    # Sort by count (descending)
    sorted_emotions = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for emotion, count in sorted_emotions:
        percentage = (count / total) * 100
        bar_length = int((count / max(counts.values())) * 25) if max(counts.values()) > 0 else 0
        bar = '█' * bar_length
        print(f"{emotion:<12} {count:<10} {percentage:>6.2f}%      {bar}")
    
    print("-"*70)
    print(f"{'TOTAL':<12} {total:<10} 100.00%")
    
    # Calculate and display imbalance
    metrics = calculate_imbalance_metrics(counts)
    if metrics:
        print(f"\n📊 Statistics:")
        print(f"   Max: {metrics['max']:<6}  Min: {metrics['min']:<6}  Mean: {metrics['mean']:.1f}  Std: {metrics['std']:.1f}")
        print(f"   Imbalance Ratio: {metrics['imbalance_ratio']:.2f}x")
        
        if metrics['imbalance_ratio'] > 3:
            print("   🔴 SEVERE IMBALANCE - Mandatory resampling needed!")
        elif metrics['imbalance_ratio'] > 2:
            print("   🟠 HIGH IMBALANCE - Resampling strongly recommended")
        elif metrics['imbalance_ratio'] > 1.5:
            print("   🟡 MODERATE IMBALANCE - Consider weighted loss")
        else:
            print("   🟢 RELATIVELY BALANCED")

def analyze_sad_happy_confusion(train_counts):
    """Specific analysis for SAD vs HAPPY confusion"""
    sad_count = train_counts.get('sad', 0)
    happy_count = train_counts.get('happy', 0)
    
    print(f"\n{'='*70}")
    print("SAD vs HAPPY CONFUSION ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n📊 Class Distribution:")
    print(f"   Happy: {happy_count} images")
    print(f"   Sad:   {sad_count} images")
    
    if happy_count > 0 and sad_count > 0:
        ratio = max(sad_count, happy_count) / min(sad_count, happy_count)
        print(f"   Ratio: {ratio:.2f}x")
        
        if ratio > 2:
            print(f"\n⚠️  IMBALANCE DETECTED!")
            print(f"   This imbalance may contribute to SAD/HAPPY confusion")
        
        print(f"\n💡 Recommended Solutions:")
        print(f"   1. ✅ Use Weighted CrossEntropyLoss")
        print(f"      - Assign higher weight to minority class")
        print(f"   2. ✅ Apply SMOTE upsampling")
        print(f"      - Create synthetic samples for minority class")
        print(f"   3. ✅ Use Focal Loss")
        print(f"      - Focus on hard-to-classify examples")
        print(f"   4. ✅ Data Augmentation")
        print(f"      - Different augmentations for SAD vs HAPPY")
        print(f"   5. ✅ Binary Classifier Layer")
        print(f"      - Add final check: is it positive (happy) or negative (sad)?")

def visualize_distribution(train_counts, val_counts, test_counts):
    """Create visualization of dataset distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = plt.cm.Set3(range(len(EMOTIONS)))
    
    # 1. Train distribution
    if sum(train_counts.values()) > 0:
        axes[0, 0].bar(train_counts.keys(), train_counts.values(), color=colors, edgecolor='black')
        axes[0, 0].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Emotion')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        for i, (emotion, count) in enumerate(train_counts.items()):
            axes[0, 0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. Val distribution
    if sum(val_counts.values()) > 0:
        axes[0, 1].bar(val_counts.keys(), val_counts.values(), color=colors, edgecolor='black')
        axes[0, 1].set_title('Validation Set Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Emotion')
        axes[0, 1].set_ylabel('Number of Images')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for i, (emotion, count) in enumerate(val_counts.items()):
            axes[0, 1].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # 3. Test distribution
    if sum(test_counts.values()) > 0:
        axes[1, 0].bar(test_counts.keys(), test_counts.values(), color=colors, edgecolor='black')
        axes[1, 0].set_title('Test Set Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Emotion')
        axes[1, 0].set_ylabel('Number of Images')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        for i, (emotion, count) in enumerate(test_counts.items()):
            axes[1, 0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # 4. Comparison across splits
    emotions_list = list(EMOTIONS)
    x = np.arange(len(emotions_list))
    width = 0.25
    
    train_vals = [train_counts.get(e, 0) for e in emotions_list]
    val_vals = [val_counts.get(e, 0) for e in emotions_list]
    test_vals = [test_counts.get(e, 0) for e in emotions_list]
    
    axes[1, 1].bar(x - width, train_vals, width, label='Train', alpha=0.8)
    axes[1, 1].bar(x, val_vals, width, label='Val', alpha=0.8)
    axes[1, 1].bar(x + width, test_vals, width, label='Test', alpha=0.8)
    
    axes[1, 1].set_title('Comparison Across Splits', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Emotion')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(emotions_list, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rafdb_dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: rafdb_dataset_analysis.png")
    plt.show()

def main():
    print("="*70)
    print("RAF-DB DATASET ANALYSIS")
    print("="*70)
    
    # Count images in each split
    print("\n📁 Scanning directories...")
    train_counts = count_images_per_emotion(TRAIN_DIR)
    val_counts = count_images_per_emotion(VAL_DIR)
    test_counts = count_images_per_emotion(TEST_DIR)
    
    # Print distributions
    print("\n" + "="*70)
    print("TRAINING SET")
    print("="*70)
    print_distribution("Train", train_counts)
    
    print("\n" + "="*70)
    print("VALIDATION SET")
    print("="*70)
    print_distribution("Validation", val_counts)
    
    print("\n" + "="*70)
    print("TEST SET")
    print("="*70)
    print_distribution("Test", test_counts)
    
    # SAD vs HAPPY analysis
    if train_counts:
        analyze_sad_happy_confusion(train_counts)
    
    # Visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    visualize_distribution(train_counts, val_counts, test_counts)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()