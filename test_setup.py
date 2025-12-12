# test_setup.py
"""
Quick test to verify VS Code setup
"""
import sys
from pathlib import Path

print("="*70)
print("TESTING RAF-DB VS CODE SETUP")
print("="*70)

checks = {
    'Python Version': sys.version_info >= (3, 8),
    'Checkpoint File': Path('checkpoints/rafdb_multi_attribute_final.pth').exists(),
    'Config File': Path('config/config.yaml').exists(),
    'Source Files': Path('src/improved_model_v2.py').exists(),
}

print("\nChecking setup...")
all_pass = True

for check_name, status in checks.items():
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check_name}")
    if not status:
        all_pass = False

# Test imports
print("\nTesting imports...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed")
    all_pass = False

try:
    import torchvision
    print(f"✅ torchvision {torchvision.__version__}")
except ImportError:
    print("❌ torchvision not installed")
    all_pass = False

try:
    from src.improved_model_v2 import RAF_DB_MultiAttribute_Net
    print("✅ Model import successful")
except Exception as e:
    print(f"❌ Model import failed: {e}")
    all_pass = False

try:
    from src.inference import EmotionPredictor
    print("✅ Inference import successful")
except Exception as e:
    print(f"❌ Inference import failed: {e}")
    all_pass = False

print("\n" + "="*70)

if all_pass:
    print("✅ ALL CHECKS PASSED!")
    print("\nYou're ready to use the system!")
    print("\nNext steps:")
    print("1. python predict_single.py path/to/image.jpg")
    print("2. python predict_batch.py data/sample_images/")
    print("3. python evaluate_test.py")
    print("4. python gradio_app.py (optional web interface)")
else:
    print("❌ SOME CHECKS FAILED")
    print("\nPlease fix the issues above before proceeding.")
    
    if not Path('checkpoints/rafdb_multi_attribute_final.pth').exists():
        print("\n⚠️  Missing model checkpoint!")
        print("   Download 'rafdb_multi_attribute_final.pth' from Kaggle")
        print("   Place it in: checkpoints/rafdb_multi_attribute_final.pth")

print("="*70)