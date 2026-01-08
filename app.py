"""
Skin Lesion Classifier - Full Featured Version
Models loaded from HuggingFace Hub on-demand
"""
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import timm
import torchvision.models as tv_models
import pandas as pd

# Page config
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem !important;
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    /* Mobile responsive - smaller fonts on phones */
    @media (max-width: 768px) {
        h1 {
            font-size: 24px !important;
        }
        h2, h3 {
            font-size: 18px !important;
        }
        p, span, div, label {
            font-size: 14px !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 12px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Constants
HF_REPO_ID = "muhwira27/skin-lesion-models"
AVAILABLE_MODELS = {
    "shufflenet_v2_x1_0": {"name": "ShuffleNet V2", "f1": 0.648, "params": "1.26M", "file": "shufflenet_v2_x1_0_best.ckpt"},
    "densenet121": {"name": "DenseNet-121", "f1": 0.637, "params": "6.96M", "file": "densenet121_best.ckpt"},
    "vit_small_patch16_224": {"name": "ViT-Small", "f1": 0.624, "params": "21.67M", "file": "vit_small_patch16_224_best.ckpt"},
    "mobilenet_v3_large": {"name": "MobileNetV3", "f1": 0.631, "params": "4.21M", "file": "mobilenet_v3_large_best.ckpt"},
    "resnet50": {"name": "ResNet-50", "f1": 0.604, "params": "23.52M", "file": "resnet50_best.ckpt"},
    "seresnet50": {"name": "SE-ResNet50", "f1": 0.617, "params": "26.05M", "file": "seresnet50_best.ckpt"},
    "tf_efficientnetv2_s": {"name": "EfficientNetV2-S", "f1": 0.615, "params": "20.19M", "file": "tf_efficientnetv2_s_best.ckpt"},
    "regnety_032": {"name": "RegNetY-032", "f1": 0.608, "params": "17.93M", "file": "regnety_032_best.ckpt"},
    "convnext_tiny": {"name": "ConvNeXt Tiny", "f1": 0.601, "params": "27.82M", "file": "convnext_tiny_best.ckpt"},
    "efficientnet_b0": {"name": "EfficientNet-B0", "f1": 0.613, "params": "4.02M", "file": "efficientnet_b0_best.ckpt"},
}

# Class descriptions for medical context (original format)
CLASS_INFO = {
    "ACK": {
        "name": "Actinic Keratosis",
        "description": "Rough, scaly patch caused by sun damage. Pre-cancerous condition.",
        "severity": "Medium",
        "color": "#FFA726"
    },
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "description": "Most common type of skin cancer. Rarely spreads but can be destructive locally.",
        "severity": "High",
        "color": "#EF5350"
    },
    "MEL": {
        "name": "Melanoma",
        "description": "Most dangerous form of skin cancer. Early detection is critical.",
        "severity": "Critical",
        "color": "#D32F2F"
    },
    "NEV": {
        "name": "Nevus (Mole)",
        "description": "Benign growth of melanocytes. Common and usually harmless.",
        "severity": "Low",
        "color": "#66BB6A"
    },
    "SCC": {
        "name": "Squamous Cell Carcinoma",
        "description": "Second most common skin cancer. Can spread if not treated.",
        "severity": "High",
        "color": "#EF5350"
    },
    "SEK": {
        "name": "Seborrheic Keratosis",
        "description": "Benign skin growth. Common in older adults, not cancerous.",
        "severity": "Low",
        "color": "#66BB6A"
    }
}

def build_model(backbone_name, num_classes):
    """Build model from backbone name."""
    if backbone_name == 'resnet50':
        model = tv_models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'densenet121':
        model = tv_models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif backbone_name == 'shufflenet_v2_x1_0':
        model = tv_models.shufflenet_v2_x1_0(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'mobilenet_v3_large':
        model = tv_models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes)
    return model

def get_target_layer(model, backbone_name):
    """Get target layer for Grad-CAM based on backbone."""
    if backbone_name == 'resnet50':
        return model.layer4[-1]
    elif backbone_name == 'densenet121':
        return model.features.denseblock4.denselayer16
    elif backbone_name == 'shufflenet_v2_x1_0':
        return model.conv5
    elif backbone_name == 'mobilenet_v3_large':
        return model.features[-1]
    elif 'efficientnet' in backbone_name:
        return model.conv_head if hasattr(model, 'conv_head') else model.blocks[-1]
    elif 'convnext' in backbone_name:
        return model.stages[-1].blocks[-1]
    elif 'vit' in backbone_name:
        return model.blocks[-1].norm1
    elif 'regnet' in backbone_name:
        return model.s4
    elif 'seresnet' in backbone_name:
        return model.layer4[-1]
    else:
        return None

class SimpleGradCAM:
    """Simple Grad-CAM implementation without external dependencies."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        if target_layer is not None:
            target_layer.register_forward_hook(self._save_activation)
            target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class):
        if self.target_layer is None:
            return None
            
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return None
        
        try:
            # Check tensor dimensions - ViT has different shape than CNNs
            if len(self.gradients.shape) == 4:
                # CNN: (B, C, H, W)
                weights = self.gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * self.activations).sum(dim=1, keepdim=True)
            elif len(self.gradients.shape) == 3:
                # ViT: (B, num_patches, hidden_dim) - e.g., (1, 197, 384)
                # For ViT, we average over patches and reshape
                weights = self.gradients.mean(dim=2, keepdim=True)  # (B, num_patches, 1)
                cam = (weights * self.activations).sum(dim=2)  # (B, num_patches)
                
                # Reshape to 2D grid (exclude CLS token if present)
                num_patches = cam.shape[1]
                if num_patches == 197:  # 14x14 + 1 CLS token
                    cam = cam[:, 1:]  # Remove CLS token
                    num_patches = 196
                
                grid_size = int(num_patches ** 0.5)
                cam = cam.view(1, grid_size, grid_size)
            else:
                return None
            
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
        except Exception:
            # If Grad-CAM fails for any reason, return None gracefully
            return None

@st.cache_resource
def load_model_from_hub(backbone_key):
    """Download and load model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    
    model_info = AVAILABLE_MODELS[backbone_key]
    filename = model_info["file"]
    
    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    backbone = checkpoint["backbone"]
    label2id = checkpoint["label2id"]
    img_size = checkpoint["img_size"]
    num_classes = len(label2id)
    id2label = {v: k for k, v in label2id.items()}
    
    model = build_model(backbone, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Setup Grad-CAM
    target_layer = get_target_layer(model, backbone)
    gradcam = SimpleGradCAM(model, target_layer)
    
    return model, img_size, id2label, backbone, gradcam

def preprocess_image(image, img_size):
    """Preprocess image for inference."""
    img = image.resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    tensor.requires_grad = True
    return tensor

def apply_colormap(cam, image):
    """Apply colormap to CAM and overlay on image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    img_resized = cv2.resize(np.array(image), (224, 224)) / 255.0
    overlay = 0.5 * heatmap + 0.5 * img_resized
    overlay = np.clip(overlay, 0, 1)
    return overlay

def main():
    # Header
    st.markdown('<h1 style="font-size: 36px; font-weight: bold; color: #1E88E5; text-align: center; padding: 15px 0;">🩺 Skin Lesion Classifier & Explainability</h1>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "📊 Benchmark Results", "ℹ️ About"])
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        sorted_models = sorted(AVAILABLE_MODELS.items(), key=lambda x: x[1]["f1"], reverse=True)
        model_options = [f"{v['name']} (F1: {v['f1']:.3f})" for k, v in sorted_models]
        model_keys = [k for k, v in sorted_models]
        
        # Find DenseNet index (default model)
        default_idx = next((i for i, k in enumerate(model_keys) if k == "densenet121"), 0)
        selected_idx = st.selectbox("Select Model", range(len(model_options)), index=default_idx, format_func=lambda i: model_options[i])
        selected_key = model_keys[selected_idx]
        selected_info = AVAILABLE_MODELS[selected_key]
        
        # Load model with progress indicator
        model_placeholder = st.empty()
        
        # Check if model is already cached
        @st.cache_data
        def is_model_cached(key):
            return key
        
        try:
            # Show loading progress
            with model_placeholder.container():
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()
                
                # Step 1: Check cache
                status_text.text("📦 Checking cache...")
                progress_bar.progress(10, text="Checking cache...")
                
                # Step 2: Download from HF Hub
                status_text.text(f"⬇️ Downloading {selected_info['name']} from HuggingFace Hub...")
                progress_bar.progress(30, text=f"Downloading {selected_info['name']}...")
                
                model, img_size, id2label, backbone, gradcam = load_model_from_hub(selected_key)
                
                # Step 3: Loading weights
                status_text.text("🔧 Loading model weights...")
                progress_bar.progress(70, text="Loading weights...")
                
                # Step 4: Setting up Grad-CAM
                status_text.text("🧠 Initializing Grad-CAM...")
                progress_bar.progress(90, text="Setting up explainability...")
                
                # Complete
                progress_bar.progress(100, text="Ready!")
                status_text.empty()
            
            # Clear progress and show success
            model_placeholder.empty()
            st.success(f"✅ Loaded: {selected_info['name']}")
            st.caption(f"Image Size: {img_size} | Params: {selected_info['params']}")
        except Exception as e:
            model_placeholder.empty()
            st.error(f"Failed to load model: {e}")
            return
        
        st.divider()
        
        # Show class legend (original format)
        st.subheader("📋 Class Legend")
        for code, info in CLASS_INFO.items():
            severity_color = info["color"]
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <span style="background-color: {severity_color}; padding: 2px 8px; border-radius: 3px; color: white; font-weight: bold; margin-right: 8px;">{code}</span>
                <span style="font-size: 0.85rem;">{info['name']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 1: Image Analysis
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader("Choose a skin lesion image", type=["jpg", "png", "jpeg"])
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if uploaded_file:
                st.subheader("🔍 Prediction Results")
                
                input_tensor = preprocess_image(image, img_size)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                sorted_idx = probs.argsort()[::-1]
                top_pred_idx = sorted_idx[0]
                top_pred_label = id2label[top_pred_idx]
                top_pred_prob = probs[top_pred_idx]
                
                info = CLASS_INFO.get(top_pred_label, {})
                severity = info.get("severity", "Unknown")
                severity_colors = {"Critical": "#D32F2F", "High": "#EF5350", "Medium": "#FFA726", "Low": "#66BB6A"}
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {info.get('color', '#1E88E5')}44 0%, {info.get('color', '#1E88E5')}22 100%); 
                            border-left: 4px solid {info.get('color', '#1E88E5')}; 
                            padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: {info.get('color', '#1E88E5')};">{top_pred_label}: {info.get('name', 'Unknown')}</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{top_pred_prob*100:.1f}%</p>
                    <p style="margin: 0; font-size: 0.9rem;">{info.get('description', '')}</p>
                    <span style="background-color: {severity_colors.get(severity, '#999')}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.8rem;">
                        Severity: {severity}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**All Class Probabilities:**")
                for idx in sorted_idx:
                    label = id2label[idx]
                    prob = probs[idx]
                    st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
                
                # Warning for BCC/SCC confusion (especially in lightweight models)
                is_lightweight = selected_key in ["shufflenet_v2_x1_0", "mobilenet_v3_large", "efficientnet_b0"]
                is_cancer_prediction = top_pred_label in ["BCC", "SCC"]
                
                if is_cancer_prediction and top_pred_prob < 0.7:
                    # Get second prediction
                    second_pred_label = id2label[sorted_idx[1]]
                    second_pred_prob = probs[sorted_idx[1]]
                    
                    # Check if BCC/SCC confusion is likely
                    if (top_pred_label == "SCC" and second_pred_label == "BCC") or \
                       (top_pred_label == "BCC" and second_pred_label == "SCC"):
                        st.warning(f"""
                        ⚠️ **Low Confidence Skin Cancer Prediction**
                        
                        The model predicts **{top_pred_label}** with only **{top_pred_prob*100:.1f}%** confidence, 
                        while **{second_pred_label}** has **{second_pred_prob*100:.1f}%**.
                        
                        BCC and SCC can be visually similar. We recommend:
                        - 🔄 Try a different model (e.g., DenseNet-121, SE-ResNet50)
                        - 👨‍⚕️ Consult a dermatologist for definitive diagnosis
                        """)
                    elif is_lightweight:
                        st.info(f"""
                        💡 **Tip:** For skin cancer predictions with confidence < 70%, 
                        consider trying a larger model like **DenseNet-121** or **SE-ResNet50** for a second opinion.
                        """)
        
        # Grad-CAM Section
        if uploaded_file:
            st.divider()
            st.subheader("🧠 Explainability (Grad-CAM)")
            
            gcol1, gcol2 = st.columns(2)
            
            with gcol1:
                # Generate Grad-CAM
                input_tensor = preprocess_image(image, img_size)
                cam = gradcam.generate(input_tensor, int(top_pred_idx))
                
                if cam is not None:
                    overlay = apply_colormap(cam, image)
                    st.image(overlay, caption=f"Grad-CAM: Areas influencing '{top_pred_label}' prediction", use_container_width=True)
                    
                    # Note for ViT models
                    if 'vit' in backbone.lower():
                        st.caption("ℹ️ *ViT uses patch-based attention (14×14 grid), so visualization may appear more blocky compared to CNN models.*")
                else:
                    st.warning("Grad-CAM not available for this model architecture")
            
            with gcol2:
                st.markdown("""
                **How to interpret Grad-CAM:**
                - 🔴 **Red/Yellow areas**: High importance for the prediction
                - � **Blue areas**: Low importance
                - The model focuses on these regions when making its decision
                
                **Clinical Note:**
                - Check if the model is focusing on the lesion itself
                - If focus is on background/artifacts, prediction may be unreliable
                """)
                
                if top_pred_prob > 0.7:
                    st.success("✅ High confidence prediction")
                elif top_pred_prob > 0.4:
                    st.warning("⚠️ Moderate confidence - consider second opinion")
                else:
                    st.error("❌ Low confidence - prediction uncertain")
    
    # Tab 2: Benchmark Results
    with tab2:
        st.subheader("📊 Benchmark Performance (5-Fold Cross-Validation)")
        
        # Performance table
        data = []
        for i, (k, v) in enumerate(sorted_models):
            data.append({
                "Rank": i + 1,
                "Model": v["name"],
                "Macro-F1": v["f1"],
                "Std Dev": [0.026, 0.027, 0.032, 0.060, 0.022, 0.035, 0.025, 0.043, 0.072, 0.089][i],
                "Params": v["params"]
            })
        df = pd.DataFrame(data)
        
        display_df = df.copy()
        display_df["Macro-F1"] = display_df["Macro-F1"].apply(lambda x: f"{x:.3f}")
        display_df["Std Dev"] = display_df["Std Dev"].apply(lambda x: f"±{x:.3f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Bar chart
        st.subheader("📈 Performance Comparison")
        chart_data = df[["Model", "Macro-F1"]].set_index("Model")
        st.bar_chart(chart_data, horizontal=True, height=400)
        
        # Key insights
        st.subheader("💡 Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="🥇 Best Model", value="ShuffleNet V2", delta="F1: 0.648")
        
        with col2:
            st.metric(label="🪶 Best Lightweight", value="ShuffleNet V2", delta="Only 1.26M params!")
        
        st.info("🔥 **Surprising Finding**: ShuffleNet V2 (ultra-lightweight) achieves the best performance!")
        
        st.info("""
💡 **Default Model:** DenseNet-121 — more robust on real-world images than ShuffleNet V2 (top CV performer).  
*Benchmark metrics may not fully reflect generalization, especially for lightweight models.*
""")
    
    # Tab 3: About
    with tab3:
        st.subheader("ℹ️ About This Project")
        st.markdown("""
        ### Skin Lesion Classification Benchmark
        
        This project is an **end-to-end deep learning benchmark** for skin lesion classification 
        using the **PAD-UFES-20** dataset.
        
        **Key Features:**
        - 🧠 **10 CNN/Transformer architectures** benchmarked
        - 📊 **5-fold cross-validation** for statistical rigor
        - � **Grad-CAM visualization** for model interpretability
        - 🚀 **HuggingFace Hub** integration for model hosting
        
        **Dataset Classes:**
        """)
        
        for code, info in CLASS_INFO.items():
            st.markdown(f"- **{code}** ({info['name']}): {info['description']}")
        
        st.markdown("""
        ---
        
        **⚠️ Disclaimer:**
        This is a research/portfolio project and is NOT intended for clinical diagnosis. 
        Always consult a qualified dermatologist for medical advice.
        
        ---
        
        **Author:** Muh. Wira Ramdhani Fadhil  
        *AI/ML Engineer & Computer Vision Enthusiast*
        
        **GitHub:** [skin-lesion-cnn-benchmark](https://github.com/muhwira27/skin-lesion-cnn-benchmark)
        """)

if __name__ == "__main__":
    main()
