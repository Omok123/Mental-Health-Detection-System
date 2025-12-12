# src/improved_model_v2.py
"""
Multi-attribute emotion recognition model
Matches the Kaggle training architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, EfficientNet_B0_Weights

class RAF_DB_MultiAttribute_Net(nn.Module):
    """
    Multi-attribute emotion recognition model
    
    Predicts:
    1. Emotion (7 classes)
    2. Valence (positive/negative)
    3. Arousal (high/low)
    4. Dominance (strong/weak)
    """
    
    def __init__(self, num_classes=7, model_name='resnet18', pretrained=False, dropout=0.5):
        super(RAF_DB_MultiAttribute_Net, self).__init__()
        
        # Backbone - FIXED: Use new weights API
        if model_name == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Shared features
        self.shared_features = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        
        # Emotion head (main task)
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Valence head
        self.valence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # Arousal head
        self.arousal_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # Dominance head
        self.dominance_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)
        
        emotion_logits = self.emotion_head(shared)
        valence_logits = self.valence_head(shared)
        arousal_logits = self.arousal_head(shared)
        dominance_logits = self.dominance_head(shared)
        
        return emotion_logits, valence_logits, arousal_logits, dominance_logits
    
    def predict_with_correction(self, x):
        """
        Prediction with multi-attribute correction
        Reduces confusion between all emotion pairs
        """
        emotion_logits, valence_logits, arousal_logits, dominance_logits = self.forward(x)
        
        # Get probabilities
        emotion_probs = F.softmax(emotion_logits, dim=1)
        valence_probs = F.softmax(valence_logits, dim=1)
        arousal_probs = F.softmax(arousal_logits, dim=1)
        dominance_probs = F.softmax(dominance_logits, dim=1)
        
        # Get predictions
        emotion_pred = torch.argmax(emotion_probs, dim=1)
        valence_pred = torch.argmax(valence_probs, dim=1)
        arousal_pred = torch.argmax(arousal_probs, dim=1)
        dominance_pred = torch.argmax(dominance_probs, dim=1)
        
        # Correction logic
        corrected_pred = emotion_pred.clone()
        
        for i in range(len(emotion_pred)):
            pred_emotion = emotion_pred[i].item()
            pred_valence = valence_pred[i].item()
            pred_arousal = arousal_pred[i].item()
            pred_dominance = dominance_pred[i].item()
            
            # Get expected attributes
            expected_attrs = self.get_emotion_attributes(pred_emotion)
            
            # Check consistency
            mismatches = 0
            if pred_valence != expected_attrs['valence']:
                mismatches += 1
            if pred_arousal != expected_attrs['arousal']:
                mismatches += 1
            if pred_dominance != expected_attrs['dominance']:
                mismatches += 1
            
            # If 2+ mismatches, find better match
            if mismatches >= 2:
                best_match = self.find_best_emotion_match(
                    pred_valence, pred_arousal, pred_dominance,
                    emotion_probs[i]
                )
                corrected_pred[i] = best_match
        
        return corrected_pred, emotion_probs, valence_probs, arousal_probs, dominance_probs
    
    @staticmethod
    def get_emotion_attributes(emotion_idx):
        """
        Get expected attributes for each emotion
        CORRECTED for RAF-DB standard mapping:
        0=surprise, 1=fear, 2=disgust, 3=happy, 4=sad, 5=angry, 6=neutral
        """
        # Valence: 0=negative, 1=positive
        # Arousal: 0=low, 1=high
        # Dominance: 0=weak, 1=strong
        
        attributes = {
            0: {'valence': 1, 'arousal': 1, 'dominance': 0},  # Surprise - positive, high arousal, weak dominance
            1: {'valence': 0, 'arousal': 1, 'dominance': 0},  # Fear - negative, high arousal, weak dominance
            2: {'valence': 0, 'arousal': 0, 'dominance': 0},  # Disgust - negative, low arousal, weak dominance
            3: {'valence': 1, 'arousal': 1, 'dominance': 1},  # Happy - positive, high arousal, strong dominance
            4: {'valence': 0, 'arousal': 0, 'dominance': 0},  # Sad - negative, low arousal, weak dominance
            5: {'valence': 0, 'arousal': 1, 'dominance': 1},  # Angry - negative, high arousal, strong dominance
            6: {'valence': 0, 'arousal': 0, 'dominance': 0},  # Neutral - neutral (coded as negative), low arousal, weak dominance
        }
        return attributes[emotion_idx]
    
    @staticmethod
    def find_best_emotion_match(valence, arousal, dominance, emotion_probs, top_k=3):
        """Find emotion that best matches the predicted attributes"""
        top_emotions = torch.topk(emotion_probs, k=top_k).indices
        
        best_score = -1
        best_emotion = top_emotions[0].item()
        
        for emotion_idx in top_emotions:
            emotion_idx = emotion_idx.item()
            expected = RAF_DB_MultiAttribute_Net.get_emotion_attributes(emotion_idx)
            
            # Calculate match score
            score = 0
            if valence == expected['valence']:
                score += 1
            if arousal == expected['arousal']:
                score += 1
            if dominance == expected['dominance']:
                score += 1
            
            # Weight by emotion probability
            score = score * emotion_probs[emotion_idx].item()
            
            if score > best_score:
                best_score = score
                best_emotion = emotion_idx
        
        return best_emotion
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path, device='cpu', model_name='resnet18'):
        """Load model from Kaggle checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get config from checkpoint
        num_classes = checkpoint.get('config', {}).get('NUM_CLASSES', 7)
        
        # Initialize model
        model = RAF_DB_MultiAttribute_Net(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=False,
            dropout=0.5
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded from {checkpoint_path}")
        if 'test_accuracy' in checkpoint:
            print(f"   Test Accuracy: {checkpoint['test_accuracy']:.2f}%")
        if 'best_val_acc' in checkpoint:
            print(f"   Best Val Accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        return model, checkpoint