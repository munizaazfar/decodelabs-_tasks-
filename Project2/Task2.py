import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score

# --- Matplotlib Plot Theme Harmonization ---
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0F172A'  
plt.rcParams['axes.facecolor'] = '#0F172A'    
plt.rcParams['text.color'] = '#F1F5F9'
plt.rcParams['axes.labelcolor'] = '#38BDF8'
plt.rcParams['xtick.color'] = '#94A3B8'
plt.rcParams['ytick.color'] = '#94A3B8'

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Unified AI Classification Core",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# --- Complete UI Visibility & Bug Fix Override via CSS ---
st.markdown("""
    <style>
    /* 1. Permanent Hide Streamlit Default Top Header & Deploy Button */
    header[data-testid="stHeader"], .stDeployButton, #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* 2. Global Application & Sidebar Background Uniformity */
    .stApp, section[data-testid="stSidebar"] {
        background-color: #0F172A !important; 
        color: #F1F5F9 !important;
    }
    
    section[data-testid="stSidebar"] {
        border-right: 1px solid #1E293B !important;
        padding-top: 2rem !important;
    }
    
    /* 3. Global Typography Reset */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #F1F5F9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* 4. Sidebar Text & Widget Label Visibility Fix */
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] p {
        color: #38BDF8 !important; 
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.5px;
    }
    
    div[data-baseweb="slider"] div {
        color: #F1F5F9 !important;
    }

    /* 5. FIX: st.code Block Background & Text Visibility Bug */
    div[data-testid="stCodeBlock"] pre {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    div[data-testid="stCodeBlock"] code, div[data-testid="stCodeBlock"] code span {
        color: #38BDF8 !important;
        font-weight: 700 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* 6. Dedicated Solid Top Header Box */
    .header-box {
        background-color: #1E293B !important;
        padding: 22px 30px !important;
        border-radius: 12px !important;
        border-left: 6px solid #38BDF8 !important;
        margin-bottom: 25px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .header-title {
        color: #FFFFFF !important;
        font-size: 32px !important;
        font-weight: 800 !important;
        margin: 0 0 4px 0 !important;
    }
    .header-subtitle {
        color: #94A3B8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    
    /* 7. Uniform Grid KPI Metric Component Blocks */
    div[data-testid="stMetric"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 15px 20px !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 13px !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #38BDF8 !important;
        font-weight: 800 !important;
    }
    
    /* 8. Premium Clean Tab Headers Customization */
    button[data-baseweb="tab"] p {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    button[aria-selected="true"] {
        border-bottom-color: #38BDF8 !important;
    }
    button[aria-selected="true"] p {
        color: #38BDF8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Top Header Section ---
st.markdown("""
    <div class="header-box">
        <h1 class="header-title">⚡ NEURAL DATA INFERENCE STUDIO</h1>
        <p class="header-subtitle">DEC0DELABS INDUSTRIAL TRAINING 2026 • UNIFIED HYPERPARAMETER TESTING SYSTEM</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING
# ==========================================
iris = load_iris()
X = iris.data  
y = iris.target  
feature_names = [name.replace("(cm)", "").strip().title() for name in iris.feature_names]
target_names = [name.capitalize() for name in iris.target_names]

df = pd.DataFrame(X, columns=feature_names)
df['Species'] = [target_names[i] for i in y]

# ==========================================
# UNIFIED SIDEBAR CONTROL STATION
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ SYSTEM CONTROLS")
    st.write("Tune operational engine states live:")
    st.markdown("---")
    
    k_value = st.slider("Neighbors Parameter (Value of K):", min_value=1, max_value=21, value=5, step=2)
    st.caption("Modifying K triggers dynamic distance calculations instantly.")
    
    st.markdown("---")
    test_size_ratio = st.slider("Validation Set Allocation Ratio:", min_value=0.10, max_value=0.50, value=0.20, step=0.05)
    
    st.markdown("---")
    st.caption("⚙️ Engine Status Indicator:")
    st.code("SYSTEM: ONLINE\nPIPELINE: READY", language="yaml")

# ==========================================
# MACHINE LEARNING EXECUTION PIPELINE
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_ratio, random_state=42, shuffle=True, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred, average='weighted')
acc = accuracy_score(y_test, y_pred)

# ==========================================
# WORKSPACE INTERACTIVE TABS
# ==========================================
tab_predict, tab_diagnostics, tab_insights = st.tabs([
    "🔮 LIVE PREDICTION INTERACTION", 
    "📈 PERFORMANCE DIAGNOSTICS & MATRIX", 
    "📊 PROXIMITY BOUNDARY VECTOR MAP"
])

# ------------------------------------------
# TAB 1: LIVE PREDICTION INF-ENGINE
# ------------------------------------------
with tab_predict:
    st.markdown("### 🧬 Real-Time Feature Matrix Simulation")
    st.write("Slide parameters below to feed dimensional plant coordinates directly into the active classifier:")
    
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sl = st.slider(f"📐 {feature_names[0]}", float(df.iloc[:,0].min()), float(df.iloc[:,0].max()), float(df.iloc[:,0].mean()))
        with col2:
            sw = st.slider(f"📐 {feature_names[1]}", float(df.iloc[:,1].min()), float(df.iloc[:,1].max()), float(df.iloc[:,1].mean()))
        with col3:
            pl = st.slider(f"📐 {feature_names[2]}", float(df.iloc[:,2].min()), float(df.iloc[:,2].max()), float(df.iloc[:,2].mean()))
        with col4:
            pw = st.slider(f"📐 {feature_names[3]}", float(df.iloc[:,3].min()), float(df.iloc[:,3].max()), float(df.iloc[:,3].mean()))

    user_sample = np.array([[sl, sw, pl, pw]])
    user_sample_scaled = scaler.transform(user_sample)
    
    pred_class_idx = knn.predict(user_sample_scaled)[0]
    pred_probabilities = knn.predict_proba(user_sample_scaled)[0]
    predicted_species_name = target_names[pred_class_idx]
    confidence_score = pred_probabilities[pred_class_idx] * 100

    st.write(" ")
    c_left, c_right = st.columns([1, 1.2])
    
    with c_left:
        st.markdown(f"""
            <div style="background-color: #1E293B; padding: 35px; border-radius: 12px; 
                        text-align: center; border: 2px solid #38BDF8; box-shadow: 0 4px 25px rgba(56,189,248,0.12);">
                <span style="letter-spacing: 1.5px; font-size: 11px; color: #38BDF8; font-weight: bold; display:block; margin-bottom:10px;">PREDICTED MATRIX TARGET</span>
                <h2 style="margin: 0 0 15px 0; font-size: 36px; font-weight: 800; color: #FFFFFF;">{predicted_species_name.upper()}</h2>
                <span style="background-color: rgba(56,189,248,0.12); border: 1px solid #38BDF8; padding: 6px 18px; border-radius: 20px; font-size: 13px; color: #38BDF8; font-weight: bold;">
                    ⚡ CONFIDENCE: {confidence_score:.1f}%
                </span>
            </div>
        """, unsafe_allow_html=True)
        
    with c_right:
        st.markdown("##### 📊 Relative Class Probability Allocation Bars")
        
        fig_prob, ax_prob = plt.subplots(figsize=(6.5, 2.3))
        ui_colors = ['#F43F5E', '#38BDF8', '#10B981']
        bars = ax_prob.barh(target_names, pred_probabilities * 100, color=ui_colors, edgecolor='#334155', height=0.45)
        
        ax_prob.set_xlim(0, 100)
        ax_prob.xaxis.grid(True, linestyle=':', alpha=0.25, color='#475569')
        ax_prob.yaxis.grid(False)
        ax_prob.spines['top'].set_visible(False)
        ax_prob.spines['right'].set_visible(False)
        ax_prob.spines['left'].set_color('#334155')
        ax_prob.spines['bottom'].set_color('#334155')
        
        for bar in bars:
            width = bar.get_width()
            ax_prob.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                        va='center', ha='left', color='#F1F5F9', fontweight='bold', fontsize=9.5)
                        
        plt.tight_layout()
        st.pyplot(fig_prob)

# ------------------------------------------
# TAB 2: MODEL DIAGNOSTICS & HEATMAPS
# ------------------------------------------
with tab_diagnostics:
    st.markdown("### 📈 Pipeline Performance Metrics")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="System Accuracy Target", value=f"{acc * 100:.1f}%")
    with m2:
        st.metric(label="Weighted F1 Evaluation Score", value=f"{f1:.4f}")
    with m3:
        st.metric(label="Inbound Metrics Rows Test Count", value=f"{X_test.shape[0]} Rows")

    st.write(" ")
    diag_col1, diag_col2 = st.columns([1, 1.2])
    
    with diag_col1:
        st.markdown("##### 🧩 Confusion Matrix Correlation Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=target_names, yticklabels=target_names, ax=ax_cm,
                    annot_kws={"size": 12, "weight": "bold", "color": "#FFFFFF"})
        plt.ylabel('TRUE SPECIMEN CLASS', fontsize=9, color='#38BDF8', fontweight='bold')
        plt.xlabel('PREDICTED NODE TARGET', fontsize=9, color='#38BDF8', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_cm)

    with diag_col2:
        st.markdown("##### 📉 Parameter Elasticity Curve (Elbow Diagnostic)")
        
        k_range = range(1, 22, 2)
        accuracies = []
        for k_test in k_range:
            knn_test = KNeighborsClassifier(n_neighbors=k_test)
            knn_test.fit(X_train_scaled, y_train)
            accuracies.append(accuracy_score(y_test, knn_test.predict(X_test_scaled)))
            
        fig_curve, ax_curve = plt.subplots(figsize=(6.8, 3.3))
        plt.plot(k_range, accuracies, color='#38BDF8', linestyle='--', marker='o', linewidth=2, label="Accuracy Profile")
        
        if k_value in k_range:
            plt.scatter(k_value, accuracies[list(k_range).index(k_value)], color='#F43F5E', s=150, zorder=5, label=f"ACTIVE K ({k_value})")
        
        plt.xlabel("Neighbor Boundary Configuration (K Value)", fontsize=9, color='#94A3B8')
        plt.ylabel("Validation Metric Accuracy Ratio", fontsize=9, color='#94A3B8')
        plt.grid(True, linestyle=':', alpha=0.25, color='#475569')
        plt.legend(loc="lower left", facecolor='#1E293B', edgecolor='#334155')
        plt.tight_layout()
        st.pyplot(fig_curve)

# ------------------------------------------
# TAB 3: CLUSTER DISTRIBUTION VISUALIZER
# ------------------------------------------
with tab_insights:
    st.markdown("### 📊 Multi-Dimensional Proximity Mapping Plane")
    st.write("Dynamic scatter coordination metrics mapping tracker:")
    
    sel_col1, sel_col2, _ = st.columns([1, 1, 2])
    with sel_col1:
        x_axis_selection = st.selectbox("X-Axis Component Projection:", options=feature_names, index=2)
    with sel_col2:
        y_axis_selection = st.selectbox("Y-Axis Component Projection:", options=feature_names, index=3)
        
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 3.8))
    
    cluster_colors = {'Setosa': '#F43F5E', 'Versicolor': '#38BDF8', 'Virginica': '#10B981'}
    sns.scatterplot(data=df, x=x_axis_selection, y=y_axis_selection, hue='Species', palette=cluster_colors, alpha=0.85, s=85, ax=ax_scatter)
    
    x_idx = feature_names.index(x_axis_selection)
    y_idx = feature_names.index(y_axis_selection)
    
    plt.scatter(user_sample[0][x_idx], user_sample[0][y_idx], color='#FFFFFF', marker='X', s=300, edgecolors='#F43F5E', linewidths=2.5, label='🎯 CURRENT SIMULATION SEED')
    
    plt.xlabel(f"{x_axis_selection} (cm)", color='#94A3B8')
    plt.ylabel(f"{y_axis_selection} (cm)", color='#94A3B8')
    plt.grid(True, linestyle=':', alpha=0.25, color='#475569')
    plt.legend(frameon=True, facecolor='#1E293B', edgecolor='#334155')
    plt.tight_layout()
    st.pyplot(fig_scatter)