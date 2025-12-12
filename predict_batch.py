# predict_batch.py
"""
Predict emotions for multiple images
"""
import sys
from pathlib import Path
import pandas as pd

from src.inference import EmotionPredictor

def main():
    print("="*70)
    print("RAF-DB BATCH PREDICTION")
    print("="*70)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python predict_batch.py <folder_path>")
        print("\nExample:")
        print("  python predict_batch.py data/sample_images")
        return
    
    folder_path = Path(sys.argv[1])
    
    if not folder_path.exists():
        print(f"\n❌ Error: Folder not found: {folder_path}")
        return
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(folder_path.glob(f'*{ext}')))
        image_paths.extend(list(folder_path.glob(f'*{ext.upper()}')))
    
    if not image_paths:
        print(f"\n❌ No images found in {folder_path}")
        return
    
    print(f"\nFound {len(image_paths)} images")
    
    # Initialize predictor
    print("\nLoading model...")
    predictor = EmotionPredictor(
        checkpoint_path='checkpoints/rafdb_multi_attribute_final.pth',
        model_name='resnet18',
        device='cpu'
    )
    
    # Predict
    print("\nProcessing images...")
    results = predictor.predict_batch(image_paths, use_correction=True)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"\n{i}. ❌ {Path(result['image_path']).name}: {result['error']}")
        else:
            print(f"\n{i}. {Path(result['image_path']).name}")
            print(f"   Emotion: {result['emotion'].upper()} ({result['confidence']:.2%})")
            print(f"   Attributes: {result['attributes']['valence']}, "
                  f"{result['attributes']['arousal']}, {result['attributes']['dominance']}")
    
    # Save to CSV
    csv_path = 'results/batch_predictions.csv'
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    for result in results:
        if 'error' not in result:
            csv_data.append({
                'image': Path(result['image_path']).name,
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'valence': result['attributes']['valence'],
                'arousal': result['attributes']['arousal'],
                'dominance': result['attributes']['dominance']
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"\n" + "="*70)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("="*70)
    print(f"Total images: {len(results)}")
    print(f"Successful predictions: {len([r for r in results if 'error' not in r])}")
    print(f"Errors: {len([r for r in results if 'error' in r])}")
    
    # Emotion distribution
    emotions = [r['emotion'] for r in results if 'error' not in r]
    if emotions:
        print("\nEmotion Distribution:")
        emotion_counts = pd.Series(emotions).value_counts()
        for emotion, count in emotion_counts.items():
            percentage = (count / len(emotions)) * 100
            print(f"  {emotion.capitalize():10s}: {count:3d} ({percentage:5.1f}%)")

if __name__ == "__main__":
    main()