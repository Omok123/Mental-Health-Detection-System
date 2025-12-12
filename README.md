# RAF-DB Emotion Recognition System v2.0

Multi-attribute emotion recognition system with state-of-the-art accuracy and confusion mitigation.

## 🎯 Features

- **Multi-Attribute Classification**: Uses 4 classification heads (Emotion, Valence, Arousal, Dominance)
- **Confusion Mitigation**: Specifically addresses SAD/HAPPY and other emotion confusions
- **High Accuracy**: 82-87% on RAF-DB test set
- **CPU Optimized**: Runs efficiently on CPU (16GB Surface compatible)
- **Easy to Use**: Simple Python scripts and optional web interface

## 📊 Model Performance

- **Test Accuracy**: ~85% (with multi-attribute correction)
- **Baseline**: ~82% (without correction)
- **SAD/HAPPY Confusion**: <10%
- **Architecture**: ResNet18 + Multi-head classification

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download project
cd RAF-DB-Emotion-Recognition

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model

Download `rafdb_multi_attribute_final.pth` from Kaggle training output and place in:
```
checkpoints/rafdb_multi_attribute_final.pth
```
### 3.  Run the analyze_rafdb_dataset.py and 
    
### 4.  organize_dataset.py in order to properly prepare the dataset for prediction and evaluation

### 5. Predict Single Image
```bash
python predict_single.py path/to/image.jpg
```

### 6. Predict Batch
```bash
python predict_batch.py data/sample_images/
```

### 7. Evaluate Test Set
```bash
python evaluate_test.py
```

### 8. Launch Web Interface (Optional)
```bash
python gradio_app.py
```
Then open http://127.0.0.1:7860 in your browser.

## 📁 Project Structure
```
RAF-DB-Emotion-Recognition/
├── checkpoints/
│   └── rafdb_multi_attribute_final.pth  (Download from Kaggle)
├── src/
│   ├── improved_model_v2.py     # Multi-attribute model
│   ├── inference.py             # Prediction interface
│   ├── evaluate.py              # Evaluation tools
│   └── utils.py                 # Utilities
├── data/
│   ├── test_organized/          # Test set (organized by emotion)
│   └── sample_images/           # Sample images for testing
├── results/
│   ├── plots/                   # Confusion matrices, metrics
│   └── evaluation_summary.json
├── config/
│   └── config.yaml
├── predict_single.py            # Single image prediction
├── predict_batch.py             # Batch prediction
├── evaluate_test.py             # Test set evaluation
├── gradio_app.py               # Web interface
├── requirements.txt
└── README.md
```

## 🎭 Supported Emotions

1. **Happy** - Positive, high arousal, strong dominance
2. **Angry** - Negative, high arousal, strong dominance
3. **Sad** - Negative, low arousal, weak dominance
4. **Fear** - Negative, high arousal, weak dominance
5. **Disgust** - Negative, low arousal, weak dominance
6. **Surprise** - Positive, high arousal, weak dominance
7. **Neutral** - Neutral, low arousal, weak dominance

## 📖 Usage Examples

### Python API
```python
from src.inference import predict_image

# Predict single image
result = predict_image('path/to/image.jpg')

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Attributes: {result['attributes']}")
```

### Batch Processing
```python
from src.inference import EmotionPredictor

predictor = EmotionPredictor('checkpoints/rafdb_multi_attribute_final.pth')

# Predict multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_paths)

for result in results:
    print(f"{result['image_path']}: {result['emotion']}")
```

### Evaluation
```python
from src.improved_model_v2 import RAF_DB_MultiAttribute_Net
from src.evaluate import ModelEvaluator, create_test_dataloader

# Load model
model, _ = RAF_DB_MultiAttribute_Net.load_from_checkpoint(
    'checkpoints/rafdb_multi_attribute_final.pth'
)

# Evaluate
evaluator = ModelEvaluator(model)
test_loader = create_test_dataloader('data/test_organized')
results = evaluator.evaluate(test_loader)

print(f"Accuracy: {results['accuracy']:.2f}%")
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:
```yaml
model:
  checkpoint_path: "checkpoints/rafdb_multi_attribute_final.pth"
  architecture: "resnet18"

evaluation:
  use_correction: true  # Enable multi-attribute correction

device: "cpu"  # or "cuda" if GPU available
```

## 📈 Multi-Attribute Approach

Our model uses 4 classification heads to reduce confusion:

1. **Emotion Head** (Main): Predicts 7 emotions
2. **Valence Head**: Positive vs Negative
3. **Arousal Head**: High vs Low energy
4. **Dominance Head**: Strong vs Weak expression

This multi-attribute approach helps distinguish:
- SAD vs HAPPY (different valence)
- FEAR vs ANGRY (different dominance)
- SURPRISE vs DISGUST (different arousal)

## 🎓 Training Details

- **Dataset**: RAF-DB (12,271 train + 3,068 test)
- **Architecture**: ResNet18 backbone + 4 classification heads
- **Loss**: Focal Loss (emotion) + CrossEntropy (attributes)
- **Class Imbalance**: Handled with weighted sampling
- **Augmentation**: Aggressive data augmentation
- **Training Time**: 2-4 hours on Kaggle GPU

## 🤝 Contributing

Improvements welcome! Areas for contribution:
- Additional model architectures
- Better data augmentation strategies
- Mobile/edge deployment optimization
- Real-time video emotion recognition

