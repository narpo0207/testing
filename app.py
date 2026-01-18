"""
Fake Job Posting Detection - Streamlit Application
UAP Pembelajaran Mesin

Aplikasi web untuk mendeteksi lowongan pekerjaan palsu menggunakan:
1. MLP (Neural Network Base - Non-pretrained)
2. TabNet (Transfer Learning Model 1 - Attention Mechanism)
3. Transformer (Transfer Learning Model 2 - Multi-head Attention)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# PyTorch & TabNet
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        font-size: 32px;
        font-weight: bold;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .legitimate-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .fraudulent-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Define TransformerBlock ====================
class TransformerBlock(layers.Layer):
    """Custom Transformer Block for Keras model loading"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

# ==================== Load Models ====================
@st.cache_resource
def load_preprocessing():
    """Load preprocessing pipeline"""
    preprocessing = joblib.load('models/preprocessing_pipeline.pkl')
    return preprocessing

@st.cache_resource
def load_mlp_model():
    """Load MLP model"""
    model = load_model('models/mlp_model.h5')
    return model

@st.cache_resource
def load_tabnet_model():
    """Load TabNet model"""
    model = TabNetClassifier()
    model.load_model('models/tabnet_model.zip')
    return model

@st.cache_resource
def load_transformer_model():
    """Load Transformer model with custom TransformerBlock"""
    model = load_model('models/transformer_model.h5', 
                      custom_objects={'TransformerBlock': TransformerBlock})
    return model

@st.cache_data
def load_dataset():
    """Load dataset"""
    df = pd.read_csv('dataset/fake_job_postings (2).csv')
    return df

@st.cache_data
def load_evaluation_results():
    """Load evaluation results"""
    with open('models/evaluation_results.json', 'r') as f:
        results = json.load(f)
    return results

@st.cache_data
def load_model_comparison():
    """Load model comparison table"""
    df = pd.read_csv('models/model_comparison.csv')
    return df

# ==================== Preprocessing Function ====================
def preprocess_input(data, preprocessing):
    """Preprocess input data for prediction"""
    # Create DataFrame if input is dict
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Get feature info
    num_cols = preprocessing['num_cols']
    cat_cols = preprocessing['cat_cols']
    label_encoders = preprocessing['label_encoders']
    
    # Handle missing values
    for col in num_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in cat_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna('unknown', inplace=True)
    
    # Encode categorical variables
    for col in cat_cols:
        if col in df.columns:
            le = label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Scale features
    scaler = preprocessing['scaler']
    X_scaled = scaler.transform(df)
    
    return X_scaled

# ==================== Prediction Function ====================
def predict(model_name, X_scaled, mlp_model, tabnet_model, transformer_model):
    """Make prediction using selected model"""
    if model_name == "MLP (Neural Network Base)":
        pred_prob = mlp_model.predict(X_scaled, verbose=0)
        pred_class = (pred_prob > 0.5).astype(int).flatten()[0]
        confidence = pred_prob[0][0] if pred_class == 1 else 1 - pred_prob[0][0]
        
    elif model_name == "TabNet (Transfer Learning 1)":
        pred_class = tabnet_model.predict(X_scaled.astype(np.float32))[0]
        pred_prob_tabnet = tabnet_model.predict_proba(X_scaled.astype(np.float32))
        confidence = pred_prob_tabnet[0][pred_class]
        
    elif model_name == "Transformer (Transfer Learning 2)":
        pred_prob = transformer_model.predict(X_scaled, verbose=0)
        pred_class = (pred_prob > 0.5).astype(int).flatten()[0]
        confidence = pred_prob[0][0] if pred_class == 1 else 1 - pred_prob[0][0]
    
    return pred_class, float(confidence)

# ==================== Main App ====================
def main():
    # Header
    st.markdown('<div class="main-header">🔍 Fake Job Posting Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deteksi Lowongan Pekerjaan Palsu menggunakan Deep Learning & Transfer Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    
    # Model selection
    st.sidebar.subheader("Pilih Model")
    model_name = st.sidebar.selectbox(
        "Model untuk Prediksi:",
        [
            "MLP (Neural Network Base)",
            "TabNet (Transfer Learning 1)",
            "Transformer (Transfer Learning 2)"
        ]
    )
    
    # Model info
    model_descriptions = {
        "MLP (Neural Network Base)": "Neural Network sederhana dengan arsitektur 128→64→32 neurons. Model baseline non-pretrained.",
        "TabNet (Transfer Learning 1)": "Model dengan attention mechanism untuk feature selection. Interpretable dan high performance.",
        "Transformer (Transfer Learning 2)": "Multi-head attention architecture untuk menangkap relasi kompleks antar fitur."
    }
    
    st.sidebar.info(f"**Info Model:**\n\n{model_descriptions[model_name]}")
    
    # Load resources
    try:
        df = load_dataset()
        preprocessing = load_preprocessing()
        mlp_model = load_mlp_model()
        tabnet_model = load_tabnet_model()
        transformer_model = load_transformer_model()
        evaluation_results = load_evaluation_results()
        comparison_df = load_model_comparison()
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dataset & Prediksi", "📈 Performa Model", "🔬 Analisis Dataset"])
    
    # ==================== Tab 1: Dataset & Prediction ====================
    with tab1:
        st.header("Dataset & Prediksi")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Preview Dataset")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.write(f"**Total Records:** {len(df)}")
            st.write(f"**Features:** {df.shape[1]} columns")
            st.write(f"**Target Variable:** fraudulent (0=Legitimate, 1=Fraudulent)")
        
        with col2:
            st.subheader("🎯 Prediksi Lowongan Pekerjaan")
            
            # Select job posting
            job_options = df.head(50).apply(
                lambda row: f"{row.name} - {row.get('title', 'No Title')[:50]}...", axis=1
            ).tolist()
            
            selected_job = st.selectbox(
                "Pilih Job Posting untuk Diprediksi:",
                job_options
            )
            
            job_idx = int(selected_job.split(' - ')[0])
            job_data = df.loc[job_idx].drop('fraudulent').to_dict()
            
            # Show job details
            st.write("**Detail Job Posting:**")
            detail_cols = ['title', 'company_profile', 'description', 'requirements', 
                          'employment_type', 'required_experience', 'required_education']
            
            for col in detail_cols:
                if col in job_data and pd.notna(job_data[col]):
                    value = str(job_data[col])[:200]
                    st.text(f"{col.replace('_', ' ').title()}: {value}...")
            
            # Prediction button
            if st.button("🔮 Prediksi Fraudulent", use_container_width=True):
                with st.spinner('Memproses prediksi...'):
                    # Preprocess
                    X_scaled = preprocess_input(job_data, preprocessing)
                    
                    # Predict
                    pred_class, confidence = predict(
                        model_name, X_scaled, 
                        mlp_model, tabnet_model, transformer_model
                    )
                    
                    # Display result
                    st.markdown("---")
                    st.subheader("📊 Hasil Prediksi")
                    
                    if pred_class == 0:
                        st.markdown(
                            f'<div class="prediction-box legitimate-box">'
                            f'✅ LEGITIMATE JOB<br>'
                            f'<span style="font-size: 20px;">Confidence: {confidence*100:.2f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box fraudulent-box">'
                            f'⚠️ FRAUDULENT JOB<br>'
                            f'<span style="font-size: 20px;">Confidence: {confidence*100:.2f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Probability distribution
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Legitimate', 'Fraudulent'],
                        y=[1-confidence if pred_class == 1 else confidence, 
                           confidence if pred_class == 1 else 1-confidence],
                        marker_color=['#28a745', '#dc3545'],
                        text=[f"{(1-confidence if pred_class == 1 else confidence)*100:.2f}%",
                              f"{(confidence if pred_class == 1 else 1-confidence)*100:.2f}%"],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title="Distribusi Probabilitas Prediksi",
                        xaxis_title="Class",
                        yaxis_title="Probability",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Actual label
                    actual_label = df.loc[job_idx, 'fraudulent']
                    st.info(f"**Actual Label:** {'Fraudulent' if actual_label == 1 else 'Legitimate'}")
    
    # ==================== Tab 2: Model Performance ====================
    with tab2:
        st.header("Performa Model")
        
        # Comparison table
        st.subheader("📊 Perbandingan Ketiga Model")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Highlight best model
        best_model = evaluation_results['best_model']
        best_acc = evaluation_results['best_accuracy']
        st.success(f"🏆 **Best Model:** {best_model} dengan accuracy {best_acc:.4f}")
        
        # Metrics visualization
        st.subheader("📈 Visualisasi Metrik")
        
        metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        models = ['MLP', 'TabNet', 'Transformer']
        
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            values = comparison_df[metric].values
            
            fig.add_trace(
                go.Bar(x=models, y=values, marker_color=colors, 
                      text=[f'{v:.4f}' for v in values], textposition='auto',
                      name=metric, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=700, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Training history
        st.subheader("📉 Training History")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**MLP Training**")
            if os.path.exists('models/mlp_history.json'):
                with open('models/mlp_history.json', 'r') as f:
                    mlp_hist = json.load(f)
                
                fig_mlp = go.Figure()
                fig_mlp.add_trace(go.Scatter(y=mlp_hist['loss'], name='Train Loss', mode='lines'))
                fig_mlp.add_trace(go.Scatter(y=mlp_hist['val_loss'], name='Val Loss', mode='lines'))
                fig_mlp.update_layout(height=300, title="MLP Loss", showlegend=True)
                st.plotly_chart(fig_mlp, use_container_width=True)
        
        with col2:
            st.write("**TabNet Training**")
            if os.path.exists('models/tabnet_history.json'):
                with open('models/tabnet_history.json', 'r') as f:
                    tabnet_hist = json.load(f)
                
                fig_tabnet = go.Figure()
                fig_tabnet.add_trace(go.Scatter(y=tabnet_hist['loss'], name='Train Loss', mode='lines'))
                fig_tabnet.add_trace(go.Scatter(y=tabnet_hist['val_accuracy'], name='Val Acc', mode='lines'))
                fig_tabnet.update_layout(height=300, title="TabNet Metrics", showlegend=True)
                st.plotly_chart(fig_tabnet, use_container_width=True)
        
        with col3:
            st.write("**Transformer Training**")
            if os.path.exists('models/transformer_history.json'):
                with open('models/transformer_history.json', 'r') as f:
                    trans_hist = json.load(f)
                
                fig_trans = go.Figure()
                fig_trans.add_trace(go.Scatter(y=trans_hist['loss'], name='Train Loss', mode='lines'))
                fig_trans.add_trace(go.Scatter(y=trans_hist['val_loss'], name='Val Loss', mode='lines'))
                fig_trans.update_layout(height=300, title="Transformer Loss", showlegend=True)
                st.plotly_chart(fig_trans, use_container_width=True)
    
    # ==================== Tab 3: Dataset Analysis ====================
    with tab3:
        st.header("Analisis Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribusi Target Variable")
            target_counts = df['fraudulent'].value_counts()
            
            fig_target = go.Figure()
            fig_target.add_trace(go.Bar(
                x=['Legitimate', 'Fraudulent'],
                y=target_counts.values,
                marker_color=['#28a745', '#dc3545'],
                text=target_counts.values,
                textposition='auto'
            ))
            fig_target.update_layout(
                title="Distribution of Fraudulent Jobs",
                xaxis_title="Class",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
            st.write(f"**Legitimate Jobs:** {target_counts[0]} ({target_counts[0]/len(df)*100:.2f}%)")
            st.write(f"**Fraudulent Jobs:** {target_counts[1]} ({target_counts[1]/len(df)*100:.2f}%)")
            st.write(f"**Class Imbalance Ratio:** {target_counts[0]/target_counts[1]:.2f}:1")
        
        with col2:
            st.subheader("📈 Missing Values")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig_missing = go.Figure()
                fig_missing.add_trace(go.Bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    marker_color='orange'
                ))
                fig_missing.update_layout(
                    title="Missing Values per Column",
                    xaxis_title="Column",
                    yaxis_title="Missing Count",
                    height=400,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values in dataset!")
        
        # Dataset statistics
        st.subheader("📊 Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)

# ==================== Run App ====================
if __name__ == "__main__":
    main()
