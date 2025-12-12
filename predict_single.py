# predict_single.py
"""
Predict emotion from a single image
"""
import sys
from pathlib import Path

from src.inference import EmotionPredictor

def main():
    print("="*70)
    print("RAF-DB SINGLE IMAGE PREDICTION")
    print("="*70)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python predict_single.py <image_path>")
        print("\nExample:")
        print("  python predict_single.py data/sample_images/happy_001.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"\n❌ Error: Image not found: {image_path}")
        return
    
    # Initialize predictor
    print("\nLoading model...")
    predictor = EmotionPredictor(
        checkpoint_path='checkpoints/rafdb_multi_attribute_final.pth',
        model_name='resnet18',
        device='cpu'
    )
    
    # Make prediction
    print(f"\nPredicting emotion for: {image_path}")
    result = predictor.predict(image_path, use_correction=True)
    
    # Display result
    print(predictor.format_prediction(result))
    
    # Show comparison
    print("\n" + "="*70)
    print("WITH vs WITHOUT Correction:")
    print("="*70)
    
    result_no_correction = predictor.predict(image_path, use_correction=False)
    
    if result['emotion'] != result_no_correction['emotion']:
        print(f"\n⚠️  Correction changed prediction!")
        print(f"   Without correction: {result_no_correction['emotion']} ({result_no_correction['confidence']:.2%})")
        print(f"   With correction:    {result['emotion']} ({result['confidence']:.2%})")
        print(f"   Attributes helped fix potential confusion!")
    else:
        print(f"\n✅ Both predictions agree: {result['emotion']}")

if __name__ == "__main__":
    main()