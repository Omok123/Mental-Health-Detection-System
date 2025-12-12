# mental_health_chatbot.py
"""
Mental Health Detection + LLM Chatbot
Combines emotion recognition with conversational AI support
"""
import os
from pathlib import Path
from datetime import datetime
import json

# For emotion detection
from src.inference import RobustEmotionPredictor

# For LLM integration - we'll use Groq (free and fast)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("⚠️  Groq not installed. Install with: pip install groq")
    GROQ_AVAILABLE = False

class MentalHealthChatbot:
    """
    AI Mental Health Support Chatbot
    Combines emotion detection with LLM conversation
    """
    
    def __init__(self, emotion_model_path='checkpoints/rafdb_multi_attribute_final.pth',
                 groq_api_key=None):
        """
        Initialize chatbot
        
        Args:
            emotion_model_path: Path to emotion recognition model
            groq_api_key: Groq API key (free at https://console.groq.com)
        """
        # Initialize emotion detector
        print("Loading emotion detection model...")
        self.emotion_predictor = RobustEmotionPredictor(
            checkpoint_path=emotion_model_path,
            model_name='resnet18',
            device='cpu',
            use_tta=True,
            n_tta=5
        )
        
        # Initialize LLM
        if groq_api_key is None:
            groq_api_key = os.environ.get('GROQ_API_KEY')
        
        if not groq_api_key:
            print("\n⚠️  No Groq API key found!")
            print("Get a free API key at: https://console.groq.com")
            print("Then set it: export GROQ_API_KEY='your-key-here'")
            self.llm_available = False
        else:
            self.client = Groq(api_key=groq_api_key)
            self.llm_available = True
            print("✅ LLM initialized (Groq)")
        
        # Conversation history
        self.conversation_history = []
        self.emotion_history = []
        
        # System prompt for mental health support
        self.system_prompt = """You are a compassionate and empathetic mental health support assistant. Your role is to:

1. Listen actively and validate the user's feelings
2. Provide supportive, non-judgmental responses
3. Offer practical coping strategies when appropriate
4. Recognize when professional help may be needed
5. Never diagnose or prescribe medication

Important guidelines:
- Be warm, empathetic, and supportive
- Use the detected emotion to inform your response
- If the user appears to be in crisis, gently suggest professional resources
- Keep responses concise but meaningful (2-4 sentences)
- Ask open-ended questions to encourage sharing
- Validate their feelings before offering advice

Remember: You are a supportive companion, not a replacement for professional mental health care."""
    
    def analyze_emotion_from_image(self, image_path):
        """
        Analyze emotion from facial image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with emotion analysis
        """
        print(f"\n🔍 Analyzing facial expression...")
        result = self.emotion_predictor.predict(image_path, use_correction=True, debug=False)
        
        if 'error' in result:
            return None
        
        # Store in history
        emotion_data = {
            'timestamp': datetime.now().isoformat(),
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'valence': result['attributes']['valence'],
            'arousal': result['attributes']['arousal']
        }
        self.emotion_history.append(emotion_data)
        
        return result
    
    def get_emotion_context(self):
        """Get context about recent emotions for LLM"""
        if not self.emotion_history:
            return "No facial emotion data available yet."
        
        recent = self.emotion_history[-1]
        context = f"Current detected emotion: {recent['emotion']} (confidence: {recent['confidence']:.0%}). "
        context += f"Emotional valence is {recent['valence']}, arousal is {recent['arousal']}."
        
        if len(self.emotion_history) > 1:
            emotions = [e['emotion'] for e in self.emotion_history[-5:]]
            context += f" Recent emotion pattern: {', '.join(emotions)}."
        
        return context
    
    def chat(self, user_message, image_path=None):
        """
        Have a conversation with emotion-aware support
        
        Args:
            user_message: User's text message
            image_path: Optional path to facial image for emotion detection
            
        Returns:
            Dictionary with response and emotion data
        """
        if not self.llm_available:
            return {
                'response': "LLM not available. Please set GROQ_API_KEY environment variable.",
                'emotion': None
            }
        
        # Analyze emotion if image provided
        emotion_result = None
        if image_path:
            emotion_result = self.analyze_emotion_from_image(image_path)
            if emotion_result and 'error' not in emotion_result:
                print(f"😊 Detected emotion: {emotion_result['emotion']} ({emotion_result['confidence']:.0%})")
        
        # Build context for LLM
        emotion_context = self.get_emotion_context()
        
        # Add user message to history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Build messages for LLM
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'system', 'content': f"Emotion context: {emotion_context}"}
        ]
        
        # Add conversation history (last 10 messages for context window)
        messages.extend(self.conversation_history[-10:])
        
        # Get LLM response
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Fast and high quality
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                'role': 'assistant',
                'content': assistant_message
            })
            
            return {
                'response': assistant_message,
                'emotion': emotion_result,
                'emotion_context': emotion_context
            }
            
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return {
                'response': "I'm having trouble responding right now. Please try again.",
                'emotion': emotion_result
            }
    
    def save_session(self, filepath='session_history.json'):
        """Save conversation and emotion history"""
        data = {
            'conversation': self.conversation_history,
            'emotions': self.emotion_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Session saved to {filepath}")
    
    def get_summary(self):
        """Get session summary with emotion insights"""
        if not self.emotion_history:
            return "No emotion data collected yet."
        
        emotions = [e['emotion'] for e in self.emotion_history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Calculate average valence
        valences = [e['valence'] for e in self.emotion_history]
        positive_count = valences.count('positive')
        valence_ratio = positive_count / len(valences) if valences else 0
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    SESSION SUMMARY                            ║
╚══════════════════════════════════════════════════════════════╝

📊 Emotion Analysis:
   Total expressions analyzed: {len(self.emotion_history)}
   Dominant emotion: {dominant_emotion} ({emotion_counts[dominant_emotion]} occurrences)
   Emotional valence: {valence_ratio:.0%} positive
   
📋 Emotion Distribution:
"""
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(emotions)) * 100
            bar = '█' * int(percentage / 5)
            summary += f"   {emotion.capitalize():10s} {count:2d} ({percentage:5.1f}%) {bar}\n"
        
        summary += f"\n💬 Conversation: {len(self.conversation_history)} messages exchanged\n"
        
        return summary


def main():
    """Interactive mental health chatbot demo"""
    print("="*70)
    print("🧠 MENTAL HEALTH SUPPORT CHATBOT")
    print("Emotion Detection + AI Conversation")
    print("="*70)
    
    # Check for API key
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        print("\n⚠️  Setup Required:")
        print("1. Get free API key: https://console.groq.com")
        print("2. Set environment variable:")
        print("   export GROQ_API_KEY='your-key-here'")
        print("\nOr pass it directly when creating the chatbot.")
        return
    
    # Initialize chatbot
    chatbot = MentalHealthChatbot()
    
    if not chatbot.llm_available:
        print("\n❌ Cannot start without LLM. Please configure API key.")
        return
    
    print("\n" + "="*70)
    print("💡 How to use:")
    print("   - Type your message to chat")
    print("   - Type 'image <path>' to analyze facial emotion")
    print("   - Type 'summary' to see session insights")
    print("   - Type 'quit' to exit")
    print("="*70)
    
    # Main chat loop
    while True:
        print("\n" + "-"*70)
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n" + "="*70)
            print(chatbot.get_summary())
            
            save = input("\nSave session? (y/n): ").lower()
            if save == 'y':
                chatbot.save_session()
            
            print("\n👋 Take care! Remember, professional help is always available if needed.")
            break
        
        elif user_input.lower() == 'summary':
            print(chatbot.get_summary())
            continue
        
        elif user_input.lower().startswith('image '):
            image_path = user_input[6:].strip()
            if Path(image_path).exists():
                result = chatbot.analyze_emotion_from_image(image_path)
                if result:
                    print(f"\n😊 Detected: {result['emotion']} ({result['confidence']:.0%})")
                    print(f"   Attributes: {result['attributes']['valence']}, "
                          f"{result['attributes']['arousal']}, {result['attributes']['dominance']}")
                else:
                    print("\n❌ Could not detect emotion in image")
            else:
                print(f"\n❌ Image not found: {image_path}")
            continue
        
        # Regular chat
        response = chatbot.chat(user_input)
        print(f"\n🤖 Support Bot: {response['response']}")


if __name__ == "__main__":
    main()