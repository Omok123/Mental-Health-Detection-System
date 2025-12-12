# organize_dataset.py
"""
Organize RAF-DB dataset from numbered folders to named emotion folders
"""
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
import random

# Correct RAF-DB emotion mapping
EMOTION_MAP = {
    '1': 'surprise',
    '2': 'fear',
    '3': 'disgust',
    '4': 'happy',
    '5': 'sad',
    '6': 'angry',
    '7': 'neutral'
}

# Configuration
SOURCE_TRAIN_DIR = r'C:\Users\Kingsley\Desktop\RAF-DB-Emotion-Recognition\data\train'
SOURCE_TEST_DIR = r'C:\Users\Kingsley\Desktop\RAF-DB-Emotion-Recognition\data\test'

TARGET_TRAIN_DIR = 'data/train_organized'
TARGET_VAL_DIR = 'data/val'
TARGET_TEST_DIR = 'data/test_organized'

VAL_SPLIT = 0.1  # 10% for validation from training data

def get_emotion_from_filename(filename):
    """
    Extract emotion label from filename
    Assumes format like: '1_image.jpg' or 'train_00001_aligned.jpg' with label in name
    """
    # Try to find emotion number (1-7) in filename
    for num in ['1', '2', '3', '4', '5', '6', '7']:
        if filename.startswith(num + '_') or f'_{num}_' in filename:
            return num
    return None

def organize_images_by_emotion(source_dir, target_base_dir, is_train=True, val_split=0.1):
    """
    Organize images into emotion-named folders
    If is_train=True, split into train and val
    """
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        print(f"   Please update SOURCE_TRAIN_DIR and SOURCE_TEST_DIR at the top of this script")
        return
    
    print(f"\n{'='*70}")
    print(f"Processing: {source_dir}")
    print(f"{'='*70}")
    
    # Collect all images by emotion
    emotion_images = {emotion: [] for emotion in EMOTION_MAP.values()}
    
    # Check if images are in numbered subfolders or directly in source_dir
    items = os.listdir(source_dir)
    
    # Case 1: Images in numbered subfolders (1/, 2/, 3/, etc.)
    if any(item in ['1', '2', '3', '4', '5', '6', '7'] for item in items):
        print("📁 Found numbered emotion folders")
        for emotion_num, emotion_name in EMOTION_MAP.items():
            emotion_folder = os.path.join(source_dir, emotion_num)
            if os.path.exists(emotion_folder):
                images = [f for f in os.listdir(emotion_folder) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                for img in images:
                    emotion_images[emotion_name].append(
                        os.path.join(emotion_folder, img)
                    )
    
    # Case 2: All images directly in source_dir with emotion in filename
    else:
        print("📁 Images in single folder, extracting emotion from filename")
        all_images = [f for f in items if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in all_images:
            emotion_num = get_emotion_from_filename(img_file)
            if emotion_num and emotion_num in EMOTION_MAP:
                emotion_name = EMOTION_MAP[emotion_num]
                emotion_images[emotion_name].append(
                    os.path.join(source_dir, img_file)
                )
    
    # Print statistics
    print(f"\n{'Emotion':<12} {'Count':<10}")
    print("-"*25)
    total_images = 0
    for emotion, images in sorted(emotion_images.items()):
        count = len(images)
        total_images += count
        print(f"{emotion:<12} {count:<10}")
    print("-"*25)
    print(f"{'Total':<12} {total_images:<10}")
    
    if total_images == 0:
        print("\n❌ No images found!")
        print("Please check:")
        print("1. The source directory path is correct")
        print("2. Images are in numbered folders (1/, 2/, etc.) or have emotion number in filename")
        return
    
    # Split train into train+val if needed
    if is_train:
        print(f"\n📊 Splitting into Train ({int((1-val_split)*100)}%) and Val ({int(val_split*100)}%)")
        
        for emotion_name, image_paths in emotion_images.items():
            if len(image_paths) == 0:
                continue
            
            # Split into train and val
            train_images, val_images = train_test_split(
                image_paths, 
                test_size=val_split, 
                random_state=42,
                shuffle=True
            )
            
            # Create train directories and copy
            train_emotion_dir = os.path.join(TARGET_TRAIN_DIR, emotion_name)
            os.makedirs(train_emotion_dir, exist_ok=True)
            
            for img_path in train_images:
                shutil.copy2(img_path, train_emotion_dir)
            
            # Create val directories and copy
            val_emotion_dir = os.path.join(TARGET_VAL_DIR, emotion_name)
            os.makedirs(val_emotion_dir, exist_ok=True)
            
            for img_path in val_images:
                shutil.copy2(img_path, val_emotion_dir)
            
            print(f"  {emotion_name:<12} Train: {len(train_images):<6} Val: {len(val_images):<6}")
    
    else:
        # Just organize test set
        print(f"\n📊 Organizing test set")
        
        for emotion_name, image_paths in emotion_images.items():
            if len(image_paths) == 0:
                continue
            
            # Create test directories and copy
            test_emotion_dir = os.path.join(TARGET_TEST_DIR, emotion_name)
            os.makedirs(test_emotion_dir, exist_ok=True)
            
            for img_path in image_paths:
                shutil.copy2(img_path, test_emotion_dir)
            
            print(f"  {emotion_name:<12} Test: {len(image_paths):<6}")

def main():
    print("="*70)
    print("RAF-DB DATASET ORGANIZER")
    print("="*70)
    print("\nThis script will:")
    print("1. Organize images from numbered folders (1-7) to emotion-named folders")
    print("2. Split training data into train (90%) and val (10%)")
    print("3. Organize test data by emotion")
    
    # Check if source directories exist
    if not os.path.exists(SOURCE_TRAIN_DIR) and not os.path.exists(SOURCE_TEST_DIR):
        print("\n" + "="*70)
        print("⚠️  CONFIGURATION NEEDED")
        print("="*70)
        print("\nPlease update the paths at the top of this script:")
        print(f"\nSOURCE_TRAIN_DIR = r'path/to/your/train/folder'")
        print(f"SOURCE_TEST_DIR = r'path/to/your/test/folder'")
        print("\nCurrent paths:")
        print(f"  SOURCE_TRAIN_DIR = {SOURCE_TRAIN_DIR}")
        print(f"  SOURCE_TEST_DIR = {SOURCE_TEST_DIR}")
        return
    
    # Process training data (will create train + val)
    if os.path.exists(SOURCE_TRAIN_DIR):
        organize_images_by_emotion(SOURCE_TRAIN_DIR, TARGET_TRAIN_DIR, is_train=True, val_split=VAL_SPLIT)
    
    # Process test data
    if os.path.exists(SOURCE_TEST_DIR):
        organize_images_by_emotion(SOURCE_TEST_DIR, TARGET_TEST_DIR, is_train=False)
    
    print("\n" + "="*70)
    print("✅ ORGANIZATION COMPLETE!")
    print("="*70)
    print(f"\nNew folder structure:")
    print(f"  {TARGET_TRAIN_DIR}/")
    print(f"    ├── surprise/")
    print(f"    ├── fear/")
    print(f"    ├── disgust/")
    print(f"    ├── happy/")
    print(f"    ├── sad/")
    print(f"    ├── angry/")
    print(f"    └── neutral/")
    print(f"\n  {TARGET_VAL_DIR}/")
    print(f"    └── (same structure)")
    print(f"\n  {TARGET_TEST_DIR}/")
    print(f"    └── (same structure)")

if __name__ == "__main__":
    main()