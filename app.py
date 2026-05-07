import streamlit as st
from PIL import Image
import io
import torch
import torchvision.transforms as T
import os

from utils.transforms import get_val_transforms
from utils.gradcam import generate_gradcam_overlay

MODEL_PATH = os.path.join('models', 'efficientnet_classifier.pth')
SEG_MODEL_PATH = os.path.join('models', 'unet_segmentation.pth')
MODEL_PATH_V2 = os.path.join('models', 'efficientnet_classifier_v2.pth')


@st.cache_resource
def load_classifier(device):
    try:
        # Prefer timm-created model if available (matches training), otherwise fallback to torchvision
        try:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
        except Exception:
            import torchvision.models as models
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)

        # Prefer v2 checkpoint if present
        ckpt_path = MODEL_PATH_V2 if os.path.exists(MODEL_PATH_V2) else MODEL_PATH
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        return None


@st.cache_resource
def load_segmentation(device):
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet('resnet34', classes=1, activation=None)
        model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None

def monte_carlo_predict(model, tensor, device, n_runs=50):
    import numpy as np
    model.train()  # keeps dropout ON to estimate uncertainty
    predictions = []
    with torch.no_grad():
        for _ in range(n_runs):
            logits = model(tensor.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predictions.append(probs)
    
    model.eval() # restore to eval mode
    
    predictions = np.array(predictions)
    mean_probs  = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    return mean_probs, uncertainty

def main():
    st.set_page_config(layout='wide', page_title='Brain Tumor Analyzer')
    st.markdown("""
    <style>
    .stApp { background-color: #0b1220; color: #e6f2ff }
    </style>
    """, unsafe_allow_html=True)

    st.title('Brain Tumor MRI Analyzer')
    st.sidebar.header('Upload')
    uploaded = st.sidebar.file_uploader('Upload MRI image', type=['png', 'jpg', 'jpeg'])
    run = st.sidebar.button('Run Pipeline')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = load_classifier(device)
    seg_model = load_segmentation(device)

    if uploaded is None:
        st.info('Please upload an MRI image via the sidebar.')
        return

    image = Image.open(io.BytesIO(uploaded.read())).convert('RGB')
    st.image(image, caption='Uploaded image', use_column_width=True)

    if run:
        st.write('Running pipeline...')
        transforms = get_val_transforms()
        input_tensor = transforms(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header('Original')
            st.image(image, use_column_width=True)

        if classifier is None:
            col2.warning('Classifier weights not found in models/ or failed to load.')
        else:
            # Predict
            inp = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = classifier(inp)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = probs.argmax()
            # Match label ordering used in dataset: glioma, meningioma, notumor, pituitary
            labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            conf = float(probs[pred])

            with col2:
                st.header('Grad-CAM')
                # find last conv layer heuristically
                target_layer = None
                for n, m in classifier.named_modules():
                    if isinstance(m, torch.nn.Conv2d):
                        target_layer = m
                if target_layer is not None:
                    from pytorch_grad_cam import GradCAM
                    from pytorch_grad_cam.utils.image import show_cam_on_image
                    import numpy as np
                    import matplotlib.pyplot as plt
                    
                    # Generate Grad-CAM
                    cam = GradCAM(model=classifier, target_layers=[target_layer])
                    grayscale_cam = cam(input_tensor=inp)[0]
                    
                    overlay, heatmap = generate_gradcam_overlay(classifier, target_layer, input_tensor, device)
                    st.image(heatmap, caption='Grad-CAM heatmap', use_column_width=True)
                    st.image(overlay, caption='Overlay', use_column_width=True)
                    
                    # Store grayscale_cam for explainability section
                    st.session_state['grayscale_cam'] = grayscale_cam
                else:
                    st.warning('Could not find conv layer for Grad-CAM')

            with col3:
                st.header('Segmentation')
                if seg_model is None:
                    st.warning('Segmentation model not found or failed to load.')
                else:
                    inp = input_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred_mask = seg_model(inp)
                        mask = torch.sigmoid(pred_mask)[0,0].cpu().numpy()
                    st.image(mask, caption='Predicted mask', use_column_width=True)

            st.markdown(f"**Prediction:** {labels[pred]} — **Confidence:** {conf:.3f}")

            # ═══════════════════════════════════════════════════════════════
            # MODEL THOUGHT PROCESS — Explainability Section
            # ═══════════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("Model Thought Process")
            
            if 'grayscale_cam' in st.session_state:
                import numpy as np
                import matplotlib.pyplot as plt
                
                grayscale_cam = st.session_state['grayscale_cam']
                class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
                
                # ──────────────────────────────────────────────────────────
                # 1. WHY THIS CLASS AND NOT OTHERS
                # ──────────────────────────────────────────────────────────
                st.markdown("#### 1. Why This Class and Not Others")
                
                # Create horizontal probability bars
                cols_prob = st.columns(len(labels))
                for idx, label in enumerate(labels):
                    with cols_prob[idx]:
                        prob_pct = probs[idx] * 100
                        st.metric(label=label, value=f"{prob_pct:.1f}%")
                
                # Detailed horizontal bar chart
                fig_prob, ax = plt.subplots(figsize=(10, 4))
                colors = ['#FF6B6B' if i == pred else '#95E1D3' for i in range(len(labels))]
                bars = ax.barh(labels, probs * 100, color=colors, edgecolor='white', linewidth=2)
                ax.set_xlabel('Probability (%)', fontsize=11, fontweight='bold')
                ax.set_title('Model Confidence Across All Classes', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 100)
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    ax.text(prob * 100 + 2, i, f'{prob*100:.1f}%', va='center', fontsize=10, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_prob)
                
                # ──────────────────────────────────────────────────────────
                # 2. WHERE THE MODEL LOOKED — Region Attention Breakdown
                # ──────────────────────────────────────────────────────────
                st.markdown("#### 2. Where the Model Looked")
                
                # Divide into 5 regions: UL, UR, LL, LR, Center
                h, w = grayscale_cam.shape
                h_half, w_half = h // 2, w // 2
                center_margin = h // 6
                
                regions = {
                    'Upper Left': grayscale_cam[:h_half, :w_half],
                    'Upper Right': grayscale_cam[:h_half, w_half:],
                    'Lower Left': grayscale_cam[h_half:, :w_half],
                    'Lower Right': grayscale_cam[h_half:, w_half:],
                    'Center': grayscale_cam[h_half-center_margin:h_half+center_margin, w_half-center_margin:w_half+center_margin]
                }
                
                region_activations = {name: np.mean(region) for name, region in regions.items()}
                top_region = max(region_activations, key=region_activations.get)
                
                # Plot region attention
                fig_region, ax = plt.subplots(figsize=(10, 5))
                region_names = list(region_activations.keys())
                region_values = list(region_activations.values())
                colors_region = ['#FF6B6B' if name == top_region else '#4ECDC4' for name in region_names]
                bars_region = ax.bar(region_names, region_values, color=colors_region, edgecolor='white', linewidth=2)
                ax.set_ylabel('Mean Attention Intensity', fontsize=11, fontweight='bold')
                ax.set_title('Brain Region Attention Distribution', fontsize=12, fontweight='bold')
                ax.set_ylim(0, max(region_values) * 1.15)
                
                # Add value labels on bars
                for bar, val in zip(bars_region, region_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                plt.xticks(rotation=15, ha='right')
                plt.tight_layout()
                st.pyplot(fig_region)
                
                # ──────────────────────────────────────────────────────────
                # 3. ATTENTION METRICS
                # ──────────────────────────────────────────────────────────
                st.markdown("#### 3. Attention Metrics")
                
                peak_attention = np.max(grayscale_cam) * 100
                mean_attention = np.mean(grayscale_cam) * 100
                focused_pixels = np.sum(grayscale_cam > 0.5)
                total_pixels = grayscale_cam.shape[0] * grayscale_cam.shape[1]
                focused_area_pct = (focused_pixels / total_pixels) * 100
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric(label="Peak Attention", value=f"{peak_attention:.1f}%")
                with metric_cols[1]:
                    st.metric(label="Mean Attention", value=f"{mean_attention:.1f}%")
                with metric_cols[2]:
                    st.metric(label="Scan Focused On", value=f"{focused_area_pct:.1f}%")
                
                # ──────────────────────────────────────────────────────────
                # 4. EXPLANATION SUMMARY
                # ──────────────────────────────────────────────────────────
                st.markdown("#### 4. Model's Reasoning")
                
                summary_text = f"""
                The model was **{conf*100:.1f}% confident** this is a **{labels[pred]}** scan, 
                primarily focusing on the **{top_region}** of the brain. 
                The model examined **{focused_area_pct:.1f}%** of the scan in detail, 
                with peak attention intensity of **{peak_attention:.1f}%**.
                """
                
                st.info(summary_text)

                # ──────────────────────────────────────────────────────────
                # 5. MONTE CARLO DROPOUT (UNCERTAINTY ESTIMATION)
                # ──────────────────────────────────────────────────────────
                st.markdown("---")
                st.subheader("Monte Carlo Dropout (Uncertainty Estimation)")
                st.markdown("We run the image through the model 50 times with Dropout enabled to estimate the model's uncertainty. Lower uncertainty indicates higher confidence in its prediction.")
                
                with st.spinner("Running Monte Carlo sampling..."):
                    mc_mean_probs, mc_uncertainty = monte_carlo_predict(classifier, input_tensor, device, n_runs=50)
                
                st.markdown("#### Results (50 runs)")
                mc_cols = st.columns(len(labels))
                for idx, (cls_name, mean_p, std_p) in enumerate(zip(labels, mc_mean_probs, mc_uncertainty)):
                    with mc_cols[idx]:
                        st.metric(
                            label=f"{cls_name}", 
                            value=f"{mean_p * 100:.1f}%",
                            delta=f"± {std_p * 100:.2f}%",
                            delta_color="off"
                        )


if __name__ == '__main__':
    main()
