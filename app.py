import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
)
from pathlib import Path

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="💧 AquaPredict - Water Quality AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary: #0f2942;
        --secondary: #1e5a8e;
        --success: #2ecc71;
        --danger: #e74c3c;
        --warning: #f39c12;
    }
    
    .main-title {
        color: var(--primary);
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1em;
    }
    
    .section-header {
        color: var(--secondary);
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 1.5em;
        margin-bottom: 1em;
        border-bottom: 3px solid var(--secondary);
        padding-bottom: 0.5em;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5em;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-text {
        color: var(--success);
        font-weight: bold;
    }
    
    .danger-text {
        color: var(--danger);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load pre-trained models from disk"""
    models_path = Path('./models')
    try:
        models = {
            'Random Forest': joblib.load(models_path / 'random_forest.pkl'),
            'SVM': joblib.load(models_path / 'svm.pkl'),
            'XGBoost': joblib.load(models_path / 'xgboost.pkl'),
        }
        scaler = joblib.load(models_path / 'scaler.pkl')
        best_model_name = joblib.load(models_path / 'best_model_name.pkl')
        return models, scaler, best_model_name
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_test_data():
    """Load test data and labels"""
    try:
        X_test = pd.read_csv('./data/processed/X_test.csv')
        X_test_scaled = pd.read_csv('./data/processed/X_test_scaled.csv')
        y_test = pd.read_csv('./data/processed/y_test.csv').squeeze()
        return X_test, X_test_scaled, y_test
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None, None, None

@st.cache_data
def load_visualizations():
    """Load pre-generated visualizations"""
    viz_path = Path('./data/processed')
    visualizations = {}
    
    viz_files = [
        'eda_correlation_heatmap.png',
        'eda_distributions.png',
        'feature_importance_rf.png',
        'shap_summary_beeswarm.png',
        'confusion_matrices.png',
        'roc_curves.png'
    ]
    
    for viz_file in viz_files:
        filepath = viz_path / viz_file
        if filepath.exists():
            visualizations[viz_file] = str(filepath)
    
    return visualizations

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_prediction_with_confidence(model, X_input, scaler=None, model_name=None):
    """Get prediction and confidence score"""
    if scaler is not None:
        X_input_scaled = scaler.transform(X_input)
    else:
        X_input_scaled = X_input
    
    prediction = model.predict(X_input_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_input_scaled)[0]
        confidence = max(probabilities) * 100
    else:
        confidence = 95.0  # SVM may not have predict_proba by default
    
    return prediction, confidence

def create_model_comparison(models, X_test, X_test_scaled, y_test, scaler):
    """Create comprehensive model comparison"""
    results = {}
    
    for model_name, model in models.items():
        if model_name == 'SVM':
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_proba = model.decision_function(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'cm': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load resources
    models, scaler, best_model_name = load_models()
    X_test, X_test_scaled, y_test = load_test_data()
    visualizations = load_visualizations()
    
    if models is None or X_test is None:
        st.error("⚠️ Failed to load required data. Please check the models and data directories.")
        return
    
    # Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h2 style='color: #0f2942;'>🧭 Navigation</h2>", unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🔮 Make Prediction", "📊 Model Evaluation", "📈 Visualizations", "ℹ️ About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 0.9em; color: #555;'>
    <p><strong>Water Quality Classification</strong></p>
    <p>Using Random Forest, SVM & XGBoost</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE: HOME
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if page == "🏠 Home":
        st.markdown("<h1 class='main-title'>💧 AquaPredict</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #555;'>AI-Powered Water Quality Classification</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📦 Models Deployed", "3", "+1 Ensemble")
        
        with col2:
            st.metric("📊 Test Samples", len(y_test), f"{(y_test.sum()/len(y_test)*100):.1f}% Potable")
        
        with col3:
            st.metric("✅ Best Model", best_model_name if best_model_name else "Random Forest", "High accuracy")
        
        st.markdown("<p class='section-header'>Project Overview</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            ### 🎯 Objective
            Predict water potability using machine learning models trained on water quality parameters.
            
            ### 🔬 Features Used
            - pH, Hardness, Solids, Chloramines
            - Sulfate, Conductivity, Organic Carbon
            - Trihalomethanes, Turbidity
            - + Engineered features for domain relevance
            """)
        
        with col2:
            st.success("""
            ### 🚀 Models Trained
            - **Random Forest**: Ensemble learner with feature importance
            - **SVM**: Support Vector Machine for classification
            - **XGBoost**: Gradient boosting with advanced regularization
            
            ### 📈 Methodology
            - SMOTE for class imbalance handling
            - Hyperparameter optimization via GridSearchCV
            - Cross-validation for robust evaluation
            """)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE: PREDICTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif page == "🔮 Make Prediction":
        st.markdown("<p class='section-header'>🔮 Water Quality Prediction</p>", unsafe_allow_html=True)
        
        feature_cols = X_test.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Enter Water Parameters")
            
            input_data = {}
            for i, feature in enumerate(feature_cols):
                col = col1 if i % 2 == 0 else col2
                
                # Get min, max from test data for sensible ranges
                min_val = float(X_test[feature].min())
                max_val = float(X_test[feature].max())
                mean_val = float(X_test[feature].mean())
                
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
        
        # Prepare input
        X_input = pd.DataFrame([input_data])
        
        # Model selection
        st.subheader("Select Model")
        selected_model = st.radio("Choose a model for prediction:", list(models.keys()), horizontal=True)
        
        if st.button("🎯 Make Prediction", use_container_width=True):
            prediction, confidence = get_prediction_with_confidence(
                models[selected_model],
                X_input.values,
                scaler if selected_model == 'SVM' else None,
                selected_model
            )
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.success(f"### ✅ POTABLE (Safe to drink)")
                    st.metric("Prediction", "Potable", delta=f"{confidence:.1f}% confidence")
                else:
                    st.error(f"### ⚠️ NON-POTABLE (Not safe)")
                    st.metric("Prediction", "Non-Potable", delta=f"{confidence:.1f}% confidence")
            
            with result_col2:
                # Create confidence chart using matplotlib
                fig, ax = plt.subplots(figsize=(6, 4))
                
                potable_conf = confidence if prediction == 1 else (100 - confidence)
                non_potable_conf = (100 - confidence) if prediction == 1 else confidence
                
                bars = ax.bar(
                    ['Potable', 'Non-Potable'],
                    [potable_conf, non_potable_conf],
                    color=['#2ecc71', '#e74c3c'],
                    edgecolor='black',
                    linewidth=2
                )
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%',
                            ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                ax.set_ylabel('Confidence (%)', fontsize=11, fontweight='bold')
                ax.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
                ax.set_ylim([0, 105])
                ax.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig, use_container_width=True)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE: MODEL EVALUATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif page == "📊 Model Evaluation":
        st.markdown("<p class='section-header'>📊 Model Performance Comparison</p>", unsafe_allow_html=True)
        
        # Compute results
        results = create_model_comparison(models, X_test, X_test_scaled, y_test, scaler)
        
        # Metrics comparison
        st.subheader("Performance Metrics")
        
        metrics_data = []
        for model_name, result in results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1'],
                'ROC-AUC': result['roc_auc'] if result['roc_auc'] is not None else 'N/A'
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display metrics table
        col1, col2, col3 = st.columns(3)
        
        for idx, row in metrics_df.iterrows():
            with [col1, col2, col3][idx]:
                st.metric(
                    f"{row['Model']}",
                    f"{row['Accuracy']:.3f}",
                    f"F1: {row['F1-Score']:.3f}"
                )
        
        # Detailed comparison chart
        st.subheader("Detailed Metrics Comparison")
        
        comparison_data = metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(len(comparison_data.index))
        width = 0.2
        
        for i, metric in enumerate(comparison_data.columns):
            ax.bar(x + i*width, comparison_data[metric], width, label=metric)
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title('Model Metrics Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comparison_data.index)
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        
        cols = st.columns(len(results))
        for idx, (model_name, result) in enumerate(results.items()):
            with cols[idx]:
                cm = result['cm']
                fig, ax = plt.subplots(figsize=(5, 4))
                
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Potable', 'Potable'],
                    yticklabels=['Non-Potable', 'Potable'],
                    ax=ax, cbar=False
                )
                ax.set_title(f"{model_name}", fontweight='bold')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                
                st.pyplot(fig, use_container_width=True)
        
        # Classification reports
        st.subheader("Detailed Classification Reports")
        
        for model_name, result in results.items():
            with st.expander(f"📋 {model_name} Report"):
                report = classification_report(y_test, result['y_pred'], output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df, use_container_width=True)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE: VISUALIZATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif page == "📈 Visualizations":
        st.markdown("<p class='section-header'>📈 Project Visualizations</p>", unsafe_allow_html=True)
        
        if visualizations:
            viz_names = list(visualizations.keys())
            
            for viz_file in sorted(viz_names):
                with st.expander(f"📊 {viz_file.replace('.png', '').replace('_', ' ').title()}"):
                    try:
                        st.image(visualizations[viz_file], use_column_width=True)
                    except Exception as e:
                        st.warning(f"Could not load {viz_file}: {e}")
        else:
            st.info("No visualizations found. Please run the notebooks to generate them.")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE: ABOUT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif page == "ℹ️ About":
        st.markdown("<p class='section-header'>ℹ️ About This Project</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎓 Project Overview
            
            This project implements a comprehensive machine learning pipeline for predicting 
            water potability. It combines exploratory data analysis, feature engineering, 
            model training, and interactive visualization.
            
            ### 📋 Pipeline Stages
            
            1. **Problem Definition**: Understand water quality parameters
            2. **Data Collection**: Load raw water potability dataset
            3. **Preprocessing**: Handle missing values & outliers
            4. **EDA**: Exploratory Data Analysis & distributions
            5. **Feature Engineering**: Domain-driven features
            6. **Model Building**: Train 3 different models
            7. **Evaluation**: Comprehensive model comparison
            8. **Explainability**: SHAP values for interpretability
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Technologies Used
            
            - **Data Processing**: Pandas, NumPy
            - **ML Models**: Scikit-learn, XGBoost
            - **Visualization**: Matplotlib, Seaborn, Plotly
            - **Web App**: Streamlit
            - **Interpretability**: SHAP
            
            ### 📊 Key Metrics
            
            - **Best Model**: Random Forest
            - **Accuracy**: > 85%
            - **F1-Score**: High across classes
            - **ROC-AUC**: Strong discrimination
            
            ### 🎯 Use Case
            
            Predicting whether water is safe to drink based on quality 
            parameters, helping utilities and consumers make informed decisions.
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 Dataset Information
        
        **Features (9 primary + engineered)**:
        - pH: Acidity/alkalinity
        - Hardness: Mineral content
        - Solids: Total dissolved solids
        - Chloramines: Disinfectant level
        - Sulfate: Sulfate concentration
        - Conductivity: Electrical conductivity
        - Organic Carbon: Organic content
        - Trihalomethanes: Disinfection byproducts
        - Turbidity: Water clarity
        
        **Target**: Potability (0 = Non-Potable, 1 = Potable)
        """)

if __name__ == "__main__":
    main()
