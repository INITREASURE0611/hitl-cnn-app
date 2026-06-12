"""
app.py — HITL-CNN Expert Review Interface
==========================================
Student:    Praise Inioluwa Akiniwumi  |  ID: 2443512
Supervisor: Dr Ihsan Mansoor
Project:    Hybrid Vision Model for Crop Disease Diagnosis

Run locally:
    pip install streamlit torch torchvision pillow numpy matplotlib
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push to GitHub → go to share.streamlit.io → deploy
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, csv, datetime
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HITL-CNN Crop Disease Diagnostic System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0F2644, #1A3A5C);
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
}
.main-header h1 { font-size: 1.8rem; margin: 0 0 0.5rem 0; }
.main-header p  { font-size: 0.9rem; opacity: 0.85; margin: 0; }

.auto-box {
    background: #EAF3DE;
    border: 2px solid #27AE60;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.expert-box {
    background: #FCE4D6;
    border: 2px solid #E67E22;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.kpi-card {
    background: #F0F4F8;
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
    border: 1px solid #D0DDE8;
}
.decision-log {
    background: #F8FBFF;
    border: 1px solid #BDD7EE;
    border-radius: 8px;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── MODEL DEFINITION ──────────────────────────────────────────────────────────
class MCDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__(); self.p = p
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)

class EfficientNetB4_HITL(nn.Module):
    def __init__(self, n_classes, dropout_p=0.1):
        super().__init__()
        base = models.efficientnet_b4(weights=None)
        self.backbone   = base.features
        self.pool       = base.avgpool
        self.mc_drop    = MCDropout(p=dropout_p)
        in_features     = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, n_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.mc_drop(x)
        return self.classifier(x)


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    checkpoint  = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    n_classes   = checkpoint['n_classes']
    model       = EfficientNetB4_HITL(n_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names


# ── INFERENCE ─────────────────────────────────────────────────────────────────
def preprocess(img_pil, size=224):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return tfm(img_pil).unsqueeze(0)

def mc_predict(model, tensor, T=50, temperature=0.8761):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(tensor)
            probs  = F.softmax(logits / temperature, dim=1)
            probs_list.append(probs.squeeze(0).numpy())
    probs_arr  = np.array(probs_list)
    mean_probs = probs_arr.mean(axis=0)
    std_probs  = probs_arr.std(axis=0)
    entropy    = -(mean_probs * np.log2(mean_probs + 1e-8)).sum()
    return mean_probs, entropy, std_probs


# ── GRAD-CAM ──────────────────────────────────────────────────────────────────
def generate_gradcam(model, tensor, target_class):
    model.eval()
    gradients, activations = [], []

    def save_grad(grad):
        gradients.append(grad)
    def fwd_hook(module, inp, out):
        activations.append(out)
        out.register_hook(save_grad)

    handle = model.backbone[-1].register_forward_hook(fwd_hook)
    logits = model(tensor)
    model.zero_grad()
    logits[0, target_class].backward()
    handle.remove()

    grad = gradients[0].squeeze(0)
    act  = activations[0].squeeze(0)
    w    = grad.mean(dim=(1, 2))
    cam  = (w[:, None, None] * act).sum(dim=0)
    cam  = F.relu(cam).detach().numpy()
    mn, mx = cam.min(), cam.max()
    if mx > mn:
        cam = (cam - mn) / (mx - mn)
    return cam

def overlay_gradcam(img_pil, cam):
    import scipy.ndimage as ndi
    img_arr = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    cam_r   = ndi.zoom(cam, (224 / cam.shape[0], 224 / cam.shape[1]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#0F2644')
    axes[0].imshow(img_arr); axes[0].axis('off')
    axes[0].set_title('Original Image', color='white', fontsize=11)
    axes[1].imshow(img_arr)
    axes[1].imshow(cam_r, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    axes[1].axis('off')
    axes[1].set_title('Grad-CAM — What the AI focused on', color='white', fontsize=11)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    cb = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label('Attention', color='white', fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    plt.tight_layout()
    return fig


# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_FILE = "hitl_decision_log.csv"

def log_decision(image_name, prediction, confidence, entropy,
                 routing, expert_decision, corrected_label=""):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(['timestamp', 'image', 'ai_prediction',
                        'confidence_%', 'entropy_bits', 'routing',
                        'expert_decision', 'corrected_label'])
        w.writerow([
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            image_name, prediction,
            f"{confidence*100:.2f}", f"{entropy:.4f}",
            routing, expert_decision, corrected_label
        ])


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 HITL-CNN Crop Disease Diagnostic System</h1>
        <p>Hybrid Vision Model · EfficientNet-B4 · MC Dropout UQ · Grad-CAM XAI<br>
        Student: 2443512 · Supervisor: Dr Ihsan Mansoor · MRes Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Placeholder_view_vector.svg/320px-Placeholder_view_vector.svg.png",
                 width=60) if False else None
        st.markdown("### ⚙️ System Settings")

        model_file = st.file_uploader(
            "Upload trained model (.pt file)",
            type=['pt', 'pth'],
            help="Upload efficientnet_b4_hitl.pt from your Kaggle output"
        )

        st.markdown("---")
        threshold = st.slider(
            "Confidence threshold θ (bits)",
            min_value=0.1, max_value=2.0,
            value=0.45, step=0.05,
            help="Predictions with entropy above this value are referred to expert review"
        )
        mc_passes = st.slider(
            "MC Dropout passes (T)",
            min_value=10, max_value=50,
            value=50, step=10,
            help="More passes = more accurate uncertainty estimate, but slower"
        )

        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {
                'total': 0, 'automated': 0, 'referred': 0,
                'accepted': 0, 'corrected': 0, 'escalated': 0
            }
        s = st.session_state.session_stats
        col1, col2 = st.columns(2)
        col1.metric("Total", s['total'])
        col2.metric("Referred", s['referred'])
        col1.metric("Automated", s['automated'])
        col2.metric("Corrected", s['corrected'])

        if s['total'] > 0:
            referral_rate = s['referred'] / s['total'] * 100
            st.progress(s['automated'] / s['total'])
            st.caption(f"Auto rate: {100 - referral_rate:.1f}% | Referral rate: {referral_rate:.1f}%")

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'rb') as f:
                st.download_button(
                    "📥 Download Decision Log (CSV)",
                    data=f,
                    file_name="hitl_decision_log.csv",
                    mime="text/csv"
                )

    # ── MAIN CONTENT ──────────────────────────────────────────────────────────
    if model_file is None:
        st.info("👈  Upload your trained model file in the sidebar to begin. "
                "The model file is `efficientnet_b4_hitl.pt` from your Kaggle output.")

        st.markdown("### How this system works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Step 1 — Upload**
            Upload a leaf photo from the field or test set. Any JPG or PNG image.
            """)
        with col2:
            st.markdown("""
            **Step 2 — AI Analysis**
            EfficientNet-B4 runs 50 MC Dropout passes to measure both the diagnosis and confidence level.
            """)
        with col3:
            st.markdown("""
            **Step 3 — HITL Routing**
            High confidence → automated diagnosis. Low confidence → expert review with Grad-CAM heatmap.
            """)
        return

    # Load model
    with st.spinner("Loading EfficientNet-B4 model..."):
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                tmp.write(model_file.read())
                tmp_path = tmp.name
            model, class_names = load_model(tmp_path)
            st.success(f"✓ Model loaded — {len(class_names)} disease classes ready")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    # Image upload
    st.markdown("### 📸 Upload Leaf Image")
    uploaded_img = st.file_uploader(
        "Choose a leaf photo",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a leaf photo to diagnose"
    )

    if uploaded_img is None:
        st.info("Upload a leaf image above to start diagnosis")
        return

    img_pil = Image.open(uploaded_img).convert('RGB')
    image_name = uploaded_img.name

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    col_img, col_results = st.columns([1, 2])

    with col_img:
        st.image(img_pil, caption="Uploaded leaf image", use_container_width=True)

    with col_results:
        with st.spinner(f"Running MC Dropout inference (T={mc_passes} passes)..."):
            tensor     = preprocess(img_pil)
            mean_probs, entropy, std_probs = mc_predict(model, tensor, T=mc_passes)

        top_idx   = np.argsort(mean_probs)[::-1][:5]
        top5      = [(class_names[i], mean_probs[i], std_probs[i]) for i in top_idx]
        prediction   = class_names[top_idx[0]]
        confidence   = mean_probs[top_idx[0]]
        disease_name = prediction.split('___')[-1].replace('_', ' ')
        crop_name    = prediction.split('___')[0].replace('_', ' ')

        # ── HITL ROUTING ──────────────────────────────────────────────────────
        st.markdown("#### HITL Routing Decision")
        if entropy < threshold:
            routing = "AUTOMATED"
            st.markdown(f"""
            <div class="auto-box">
                <h3 style="color:#27AE60;margin:0">✅ HIGH CONFIDENCE — AUTOMATED DIAGNOSIS</h3>
                <p style="color:#1A5E2A;margin:0.5rem 0 0 0">No expert review required</p>
                <p style="font-size:0.85rem;color:#595959">
                    Entropy {entropy:.4f} bits &lt; threshold θ={threshold} bits
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            routing = "EXPERT_REVIEW"
            st.markdown(f"""
            <div class="expert-box">
                <h3 style="color:#E67E22;margin:0">⚠️ LOW CONFIDENCE — EXPERT REVIEW NEEDED</h3>
                <p style="color:#7D3C00;margin:0.5rem 0 0 0">Please review the Grad-CAM heatmap below</p>
                <p style="font-size:0.85rem;color:#595959">
                    Entropy {entropy:.4f} bits &gt; threshold θ={threshold} bits
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── PREDICTION RESULTS ─────────────────────────────────────────────
        st.markdown("#### Prediction Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Disease", disease_name[:20])
        c2.metric("Crop", crop_name)
        c3.metric("Confidence", f"{confidence*100:.1f}%")
        c4.metric("Entropy", f"{entropy:.4f} bits")

        # Confidence bar
        conf_color = "#27AE60" if confidence > 0.7 else ("#E67E22" if confidence > 0.4 else "#C00000")
        st.progress(float(confidence))

        # Top 5 predictions
        st.markdown("#### Top 5 Predictions")
        for i, (cls, prob, std) in enumerate(top5):
            name = cls.split('___')[-1].replace('_', ' ')
            crop = cls.split('___')[0]
            bar_val = float(prob)
            cols = st.columns([3, 1, 1, 4])
            cols[0].write(f"{'🥇' if i==0 else f'{i+1}.'} {name}")
            cols[1].write(crop)
            cols[2].write(f"{prob*100:.1f}%")
            cols[3].progress(bar_val)

    # ── GRAD-CAM (only for expert review) ─────────────────────────────────────
    if routing == "EXPERT_REVIEW":
        st.markdown("---")
        st.markdown("### 🔥 Grad-CAM Explainability — What the AI Focused On")
        st.caption("The heatmap shows which regions of the leaf the AI used to make its prediction. "
                   "Red/yellow areas = high attention. Blue areas = low attention.")

        with st.spinner("Generating Grad-CAM heatmap..."):
            try:
                cam = generate_gradcam(model, tensor, top_idx[0])
                fig = overlay_gradcam(img_pil, cam)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Grad-CAM generation failed: {e}")

    # ── EXPERT DECISION PANEL ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 👨‍🌾 Expert Decision")

    if routing == "AUTOMATED":
        st.info(f"This case was automatically diagnosed as **{disease_name}** "
                f"with {confidence*100:.1f}% confidence. "
                f"No review is needed, but you can override if you disagree.")

    decision_col1, decision_col2, decision_col3 = st.columns(3)

    with decision_col1:
        accept = st.button(
            f"✅ Accept AI Prediction\n\n{disease_name}",
            use_container_width=True,
            type="primary" if routing == "EXPERT_REVIEW" else "secondary"
        )

    with decision_col2:
        correct_btn = st.button(
            "✏️ Provide Correct Diagnosis",
            use_container_width=True
        )

    with decision_col3:
        escalate = st.button(
            "⚑ Escalate to Specialist",
            use_container_width=True
        )

    # Correction input
    if 'show_correction' not in st.session_state:
        st.session_state.show_correction = False
    if correct_btn:
        st.session_state.show_correction = True

    corrected_label = ""
    if st.session_state.show_correction:
        st.markdown("**Enter the correct disease class name:**")
        corrected_label = st.text_input(
            "Correct diagnosis",
            placeholder="e.g. Tomato___Late_blight  or  Apple___healthy",
            help="Use the exact class name format: Crop___Disease"
        )
        confirm_correction = st.button("Confirm Correction", type="primary")
    else:
        confirm_correction = False

    # Process decision
    expert_decision = None
    if accept:
        expert_decision = "accepted"
        st.success(f"✅ Decision recorded: Accepted AI prediction — {disease_name}")
        st.session_state.show_correction = False
    elif confirm_correction and corrected_label:
        expert_decision = "corrected"
        st.success(f"✏️ Decision recorded: Corrected to — {corrected_label}")
        st.session_state.show_correction = False
    elif escalate:
        expert_decision = "escalated"
        st.warning("⚑ Case escalated to specialist pathologist")
        st.session_state.show_correction = False

    if expert_decision:
        log_decision(image_name, prediction, confidence, entropy,
                     routing, expert_decision, corrected_label)
        s = st.session_state.session_stats
        s['total'] += 1
        if routing == "AUTOMATED":
            s['automated'] += 1
        else:
            s['referred'] += 1
        if expert_decision == "accepted":
            s['accepted'] += 1
        elif expert_decision == "corrected":
            s['corrected'] += 1
        elif expert_decision == "escalated":
            s['escalated'] += 1
        st.rerun()

    # ── UNCERTAINTY DETAIL ─────────────────────────────────────────────────────
    with st.expander("📊 Detailed Uncertainty Analysis"):
        fig2, ax = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor('#0F2644')
        ax.set_facecolor('#1A3A5C')
        names  = [class_names[i].split('___')[-1][:20] for i in top_idx]
        probs5 = [mean_probs[i] for i in top_idx]
        stds5  = [std_probs[i]  for i in top_idx]
        ax.barh(names[::-1], probs5[::-1], xerr=stds5[::-1],
                color='#5BA3E0', edgecolor='white', alpha=0.85, capsize=4)
        ax.set_xlabel("Probability", color='white')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#2E74B5')
        ax.set_title(f"Top 5 Predictions with Uncertainty  |  Entropy: {entropy:.4f} bits",
                     color='white', fontsize=10)
        st.pyplot(fig2)
        plt.close()
        st.caption("Error bars show standard deviation across 50 MC Dropout passes — "
                   "larger bars indicate higher model uncertainty for that class.")


if __name__ == "__main__":
    main()
