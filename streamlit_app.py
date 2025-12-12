# streamlit_app.py
"""
Mental Health Detection & Support System
Streamlit Web Application
"""
import streamlit as st
import os
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import tempfile
import atexit
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from src.inference import EmotionPredictor

# Try to import Groq for LLM
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Get API key from environment or Streamlit secrets
GROQ_API_KEY = os.getenv('GROQ_API_KEY') or st.secrets.get("GROQ_API_KEY", None)

# Page configuration
st.set_page_config(
    page_title="Mental Health Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact layout
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        padding-bottom: 1rem;
        margin-bottom: 0.5rem;
    }
    .emotion-card {
        padding: 0.8rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 0.3rem 0;
        text-align: center;
    }
    .insight-box {
        padding: 0.8rem;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .tips-box {
        padding: 0.8rem;
        border-radius: 8px;
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .chat-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        background: #fafafa;
        max-height: 400px;
        overflow-y: auto;
    }
    .compact-metric {
        text-align: center;
        padding: 0.5rem;
        background: #f5f5f5;
        border-radius: 5px;
        margin: 0.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
    section.main > div {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stExpander {
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Emotion insights database
EMOTION_INSIGHTS = {
    'happy': {
        'description': 'You appear to be experiencing happiness! This positive emotion is characterized by feelings of joy, contentment, and well-being.',
        'characteristics': [
            'Smiling and relaxed facial muscles',
            'Positive thoughts and optimism',
            'Boosts immune system and health',
            'Encourages social bonding'
        ],
        'tips': [
            'Savor this moment and practice gratitude',
            'Share your joy with others',
            'Engage in positive activities',
            'Document happy moments'
        ],
        'color': '#4CAF50',
        'icon': '😊'
    },
    'sad': {
        'description': 'You appear to be experiencing sadness. This is a natural response to loss, disappointment, or difficult situations.',
        'characteristics': [
            'Feeling down or low in mood',
            'Decreased energy or motivation',
            'May include crying or withdrawal',
            'Normal human emotional experience'
        ],
        'tips': [
            'Allow yourself to feel the emotion',
            'Reach out to supportive people',
            'Engage in gentle self-care',
            'Seek professional help if persists'
        ],
        'color': '#2196F3',
        'icon': '😢'
    },
    'angry': {
        'description': 'You appear to be experiencing anger, often arising from perceived injustice, frustration, or threat.',
        'characteristics': [
            'Increased heart rate and tension',
            'Frowning or clenched jaw',
            'Signal that boundaries are crossed',
            'Provides energy to address problems'
        ],
        'tips': [
            'Take deep breaths to calm down',
            'Express feelings constructively',
            'Physical activity releases tension',
            'Identify the root cause'
        ],
        'color': '#F44336',
        'icon': '😠'
    },
    'fear': {
        'description': 'You appear to be experiencing fear or anxiety. This emotion alerts us to potential threats.',
        'characteristics': [
            'Heightened alertness and vigilance',
            'Increased heart rate or sweating',
            'Worry about future events',
            'Triggers fight-or-flight response'
        ],
        'tips': [
            'Practice grounding (5-4-3-2-1 method)',
            'Challenge anxious thoughts',
            'Use progressive muscle relaxation',
            'Seek help if overwhelming'
        ],
        'color': '#9C27B0',
        'icon': '😨'
    },
    'surprise': {
        'description': 'You appear to be experiencing surprise! This brief emotion occurs with something unexpected.',
        'characteristics': [
            'The briefest of all emotions',
            'Raised eyebrows and widened eyes',
            'Quickly transitions to other emotions',
            'Helps focus on unexpected events'
        ],
        'tips': [
            'Process the unexpected information',
            'Stay open to new experiences',
            'Use as learning opportunity',
            'Embrace with curiosity'
        ],
        'color': '#FF9800',
        'icon': '😲'
    },
    'disgust': {
        'description': 'You appear to be experiencing disgust. This emotion protects us from harmful or unpleasant things.',
        'characteristics': [
            'Wrinkling of the nose',
            'Feelings of revulsion',
            'Serves as protective mechanism',
            'May involve moral aversion'
        ],
        'tips': [
            'Identify what triggered this feeling',
            'Set healthy boundaries',
            'Practice acceptance when needed',
            'Focus on what you can control'
        ],
        'color': '#795548',
        'icon': '🤢'
    },
    'neutral': {
        'description': 'You appear to be in a neutral emotional state. This balanced state reflects emotional stability.',
        'characteristics': [
            'Represents emotional equilibrium',
            'Relaxed facial features',
            'Indicates contentment or peace',
            'Allows clear thinking'
        ],
        'tips': [
            'Maintain through regular self-care',
            'Use for reflection and planning',
            'Practice mindfulness',
            'Appreciate this peaceful moment'
        ],
        'color': '#9E9E9E',
        'icon': '😐'
    }
}

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'llm_client' not in st.session_state:
    # Auto-initialize if API key is available
    if GROQ_API_KEY and GROQ_AVAILABLE:
        try:
            st.session_state.llm_client = Groq(api_key=GROQ_API_KEY)
        except:
            st.session_state.llm_client = None
    else:
        st.session_state.llm_client = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp(prefix="streamlit_emotions_")
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

def cleanup_temp_files():
    """Clean up all temporary files on exit"""
    try:
        if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
            import shutil
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        
        for temp_file in glob.glob("temp_*.jpg"):
            try:
                os.remove(temp_file)
            except:
                pass
    except Exception as e:
        pass

atexit.register(cleanup_temp_files)

@st.cache_resource
def load_emotion_model():
    """Load emotion detection model (cached)"""
    try:
        predictor = EmotionPredictor(
            checkpoint_path='checkpoints/rafdb_multi_attribute_final.pth',
            model_name='resnet18',
            device='cpu'
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def initialize_llm(api_key):
    """Initialize Groq LLM"""
    if not GROQ_AVAILABLE:
        return None
    
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        return None

def get_llm_response(client, user_message, emotion_context, chat_history):
    """Get response from LLM"""
    system_prompt = """You are a compassionate mental health support assistant. Keep responses concise (2-3 sentences), warm, and supportive."""
    
    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'system', 'content': f"Emotion: {emotion_context}"}
        ]
        
        messages.extend(chat_history[-4:])
        messages.append({'role': 'user', 'content': user_message})
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_emotion_insights(emotion, confidence, attributes):
    """Generate insights about detected emotion"""
    if emotion not in EMOTION_INSIGHTS:
        emotion = 'neutral'
    
    insight = EMOTION_INSIGHTS[emotion]
    
    return {
        'title': f"{insight['icon']} {emotion.upper()}",
        'confidence': confidence,
        'description': insight['description'],
        'characteristics': insight['characteristics'],
        'tips': insight['tips'],
        'color': insight['color'],
        'attributes': attributes
    }

def create_compact_chart(probabilities):
    """Create compact emotion probability chart"""
    emotions = list(probabilities.keys())
    probs = [probabilities[e] * 100 for e in emotions]
    
    colors = {
        'happy': '#4CAF50', 'surprise': '#FF9800', 'neutral': '#9E9E9E',
        'sad': '#2196F3', 'angry': '#F44336', 'fear': '#9C27B0', 'disgust': '#795548'
    }
    
    bar_colors = [colors.get(e, '#757575') for e in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs, y=emotions, orientation='h',
            marker=dict(color=bar_colors),
            text=[f'{p:.0f}%' for p in probs],
            textposition='auto',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=5, b=5),
        showlegend=False,
        xaxis_title="",
        yaxis_title="",
        font=dict(size=10)
    )
    
    return fig

def process_uploaded_image(uploaded_file):
    """Process uploaded image and return prediction result"""
    temp_path = None
    try:
        image = Image.open(uploaded_file)
        temp_path = os.path.join(st.session_state.temp_dir, f"emotion_{datetime.now().timestamp()}.jpg")
        image.save(temp_path)
        
        result = st.session_state.predictor.predict(temp_path, use_correction=True)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result, None
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return None, str(e)

def main():
    # Compact Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Emotion Recognition & Support</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.session_state.predictor is None:
            with st.spinner("Loading model..."):
                st.session_state.predictor = load_emotion_model()
        
        if st.session_state.predictor:
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model failed")
            st.stop()
        
        st.markdown("---")
        st.subheader("📊 Stats")
        st.metric("Analyzed", len(st.session_state.emotion_history))
        st.metric("Messages", len(st.session_state.chat_history) // 2)
        
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.emotion_history = []
            st.session_state.chat_history = []
            st.session_state.current_result = None
            cleanup_temp_files()
            st.rerun()
        
        if st.session_state.emotion_history:
            st.markdown("---")
            session_data = {
                'emotions': st.session_state.emotion_history,
                'chat': st.session_state.chat_history,
                'timestamp': datetime.now().isoformat()
            }
            
            st.download_button(
                label="📥 Export",
                data=json.dumps(session_data, indent=2),
                file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Main Tabs
    tab1, tab2 = st.tabs(["📸 Emotion Detection & Insights", "📈 Analytics"])
    
    # Tab 1: Emotion Detection - COMPACT LAYOUT
    with tab1:
        # Top Row: Upload and Results
        top_col1, top_col2, top_col3 = st.columns([1, 1, 1])
        
        with top_col1:
            st.markdown("### 📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose image",
                type=['jpg', 'jpeg', 'png'],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                if st.button("🔍 Analyze", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        result, error = process_uploaded_image(uploaded_file)
                        
                        if error:
                            st.error(f"❌ {error}")
                        elif 'error' in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            result['timestamp'] = datetime.now().strftime("%H:%M:%S")
                            st.session_state.emotion_history.append(result)
                            st.session_state.current_result = result
                            st.rerun()
        
        # Display results if available
        if st.session_state.current_result:
            result = st.session_state.current_result
            insights = get_emotion_insights(result['emotion'], result['confidence'], result['attributes'])
            
            with top_col2:
                st.markdown("### 🎯 Detection Results")
                
                # Compact emotion card
                st.markdown(f"""
                <div class="emotion-card">
                    <h2 style="margin:0; font-size:1.8rem;">{insights['title']}</h2>
                    <h1 style="margin:3px 0; font-size:2rem;">{insights['confidence']:.0%}</h1>
                    <p style="margin:0; font-size:0.85rem; opacity:0.9;">Confidence</p>
                </div>
                """, unsafe_allow_html=True)
                
                # VAD Metrics - removed, shown in chart instead
                
                # Compact chart
                st.plotly_chart(create_compact_chart(result['all_probabilities']), use_container_width=True, config={'displayModeBar': False})
            
            with top_col3:
                st.markdown("### 💬 AI Chat")
                
                # Check if API key is configured
                if st.session_state.llm_client:
                    # Compact chat display
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    
                    if st.session_state.chat_history:
                        for msg in st.session_state.chat_history[-6:]:  # Last 6 messages only
                            if msg['role'] == 'assistant':  # Only show bot responses
                                st.markdown(f"🤖 {msg['content']}", unsafe_allow_html=True)
                    else:
                        st.info("💡 Ask me about your emotions")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Chat form with inline button
                    with st.form(key="chat_form", clear_on_submit=True):
                        user_msg = st.text_input("Your message", placeholder="How are you feeling?", label_visibility="collapsed", key="chat_input_field")
                        submitted = st.form_submit_button("➤", use_container_width=True)
                        
                        if submitted and user_msg:
                            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
                            
                            emotion_ctx = f"{result['emotion']} ({result['confidence']:.0%})"
                            bot_response = get_llm_response(st.session_state.llm_client, user_msg, emotion_ctx, st.session_state.chat_history)
                            
                            st.session_state.chat_history.append({'role': 'assistant', 'content': bot_response})
                            st.rerun()
                else:
                    st.warning("⚠️ AI Chat unavailable")
                    st.info("Configure GROQ_API_KEY in .env file or Streamlit secrets")
        
        # Bottom Row: Insights (only if result exists)
        if st.session_state.current_result:
            st.markdown("---")
            
            bottom_col1, bottom_col2 = st.columns(2)
            
            with bottom_col1:
                st.markdown("### 🧠 Understanding This Emotion")
                st.write(insights['description'])
                
                st.markdown("**Key Characteristics:**")
                for char in insights['characteristics']:
                    st.markdown(f"• {char}")
            
            with bottom_col2:
                st.markdown("### 💡 Helpful Strategies")
                for i, tip in enumerate(insights['tips'], 1):
                    st.markdown(f"**{i}.** {tip}")
            
            # Recent detections removed per user request
    
    # Tab 2: Analytics
    with tab2:
        if not st.session_state.emotion_history:
            st.info("📸 Analyze images to see analytics")
        else:
            # Compact analytics
            col1, col2 = st.columns(2)
            
            with col1:
                emotions = [e['emotion'] for e in st.session_state.emotion_history]
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                fig = px.pie(
                    values=list(emotion_counts.values()),
                    names=list(emotion_counts.keys()),
                    title="Emotion Distribution"
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                valences = [e['attributes']['valence'] for e in st.session_state.emotion_history]
                valence_counts = {'positive': valences.count('positive'), 'negative': valences.count('negative')}
                
                fig = px.bar(
                    x=list(valence_counts.keys()),
                    y=list(valence_counts.values()),
                    title="Valence Distribution",
                    color=list(valence_counts.keys()),
                    color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336'}
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                dominant = max(emotion_counts, key=emotion_counts.get)
                st.metric("Dominant", dominant.capitalize())
            
            with stat_col2:
                avg_conf = sum(e['confidence'] for e in st.session_state.emotion_history) / len(st.session_state.emotion_history)
                st.metric("Avg Confidence", f"{avg_conf:.0%}")
            
            with stat_col3:
                pos_ratio = valences.count('positive') / len(valences) if valences else 0
                st.metric("Positive", f"{pos_ratio:.0%}")
            
            with stat_col4:
                st.metric("Total", len(st.session_state.emotion_history))

if __name__ == "__main__":
    main()