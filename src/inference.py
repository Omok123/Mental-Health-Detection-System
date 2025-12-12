# src/inference.py
"""
Inference module for emotion prediction
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from src.improved_model_v2 import RAF_DB_MultiAttribute_Net

class EmotionPredictor:
    """Emotion prediction with multi-attribute model"""
    
    def __init__(self, checkpoint_path, model_name='resnet18', device='cpu'):
        """
        Initialize predictor
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_name: Model architecture
            device: Device to run inference on
        """
        self.device = device
        self.model_name = model_name
        
        # Load model
        self.model, self.checkpoint = RAF_DB_MultiAttribute_Net.load_from_checkpoint(
            checkpoint_path, device, model_name
        )
        
        # Emotion labels - CORRECTED to match RAF-DB standard and config.yaml
        self.emotions = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        
        # Attribute labels
        self.valence_labels = ['positive', 'negative']
        self.arousal_labels = ['low', 'high']
        self.dominance_labels = ['weak', 'strong']
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Predictor ready! Emotions: {self.emotions}")
    
    def predict(self, image_path, use_correction=True):
        """
        Predict emotion from image
        
        Args:
            image_path: Path to image file
            use_correction: Use multi-attribute correction
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            if use_correction:
                # Use corrected prediction
                corrected_pred, emotion_probs, valence_probs, arousal_probs, dominance_probs = \
                    self.model.predict_with_correction(image_tensor)
                
                emotion_idx = corrected_pred[0].item()
                confidence = emotion_probs[0][emotion_idx].item()
                
                valence_idx = torch.argmax(valence_probs[0]).item()
                arousal_idx = torch.argmax(arousal_probs[0]).item()
                dominance_idx = torch.argmax(dominance_probs[0]).item()
            else:
                # Standard prediction
                emotion_logits, valence_logits, arousal_logits, dominance_logits = \
                    self.model(image_tensor)
                
                emotion_probs = F.softmax(emotion_logits, dim=1)
                emotion_idx = torch.argmax(emotion_probs[0]).item()
                confidence = emotion_probs[0][emotion_idx].item()
                
                valence_idx = torch.argmax(F.softmax(valence_logits, dim=1)[0]).item()
                arousal_idx = torch.argmax(F.softmax(arousal_logits, dim=1)[0]).item()
                dominance_idx = torch.argmax(F.softmax(dominance_logits, dim=1)[0]).item()
        
        # Build result
        result = {
            'image_path': str(image_path),
            'emotion': self.emotions[emotion_idx],
            'emotion_idx': emotion_idx,
            'confidence': confidence,
            'attributes': {
                'valence': self.valence_labels[valence_idx],
                'arousal': self.arousal_labels[arousal_idx],
                'dominance': self.dominance_labels[dominance_idx]
            },
            'all_probabilities': {
                self.emotions[i]: emotion_probs[0][i].item()
                for i in range(len(self.emotions))
            }
        }
        
        return result
    
    def predict_batch(self, image_paths, use_correction=True):
        """Predict emotions for multiple images"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, use_correction)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({'image_path': str(image_path), 'error': str(e)})
        
        return results
    
    def format_prediction(self, result):
        """Format prediction as readable string"""
        if 'error' in result:
            return f"❌ Error: {result['error']}"
        
        output = f"""
╔══════════════════════════════════════════════════════════════╗
║            RAF-DB EMOTION PREDICTION                          ║
╚══════════════════════════════════════════════════════════════╝

IMAGE: {Path(result['image_path']).name}

🎯 PREDICTED EMOTION: {result['emotion'].upper()}
📊 CONFIDENCE: {result['confidence']:.2%}

📋 ATTRIBUTES:
   Valence:   {result['attributes']['valence'].upper()}
   Arousal:   {result['attributes']['arousal'].upper()}
   Dominance: {result['attributes']['dominance'].upper()}

📈 ALL PROBABILITIES:
"""
        
        # Sort probabilities
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for emotion, prob in sorted_probs:
            bar_length = int(prob * 40)
            bar = '█' * bar_length
            output += f"   {emotion.capitalize():10s} {prob:6.2%} {bar}\n"
        
        return output

def predict_image(image_path, model_checkpoint='checkpoints/rafdb_multi_attribute_final.pth', 
                  device='cpu'):
    """
    Quick function to predict single image
    
    Args:
        image_path: Path to image
        model_checkpoint: Path to model checkpoint
        device: Device to use
    
    Returns:
        Prediction result dictionary
    """
    predictor = EmotionPredictor(model_checkpoint, device=device)
    result = predictor.predict(image_path)
    return result

