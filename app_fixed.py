
import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from plant_disease_classifier import PlantDiseaseModel, predict_image

warnings.filterwarnings("ignore")

# --------- Streamlit page config (must be near top and only once) ---------
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🪴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------- Optional CSS ---------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2E7D32;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .prediction-header {
        font-size: 1.75rem;
        color: #2E7D32;
        margin-top: 0.5rem;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #ddd;
    }
    .chart-container {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --------- Cached loaders ---------
@st.cache_resource
def load_model_resources():
    """Load model, transforms, label encoder, class names, and config."""
    # Always use OS-agnostic paths
    with open(os.path.join("results", "model_config.json"), "r") as f:
        config = json.load(f)

    with open(config["class_names_path"], "r") as f:
        class_names = json.load(f)

    with open(config["label_encoder_path"], "rb") as f:
        label_encoder = pickle.load(f)

    with open(config["inference_transform_path"], "rb") as f:
        transform = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes=len(class_names))
    # If you're on PyTorch >=2.0 you can set weights_only=True; otherwise default works.
    try:
        state = torch.load(config["model_path"], map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(config["model_path"], map_location=device)

    # Handle both raw state_dict and checkpoints with 'state_dict'
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, transform, label_encoder, class_names, device, config


# --------- Helpers ---------
def format_class_name(name: str) -> str:
    return name.replace("_", " ").replace("__", " ").replace("___", " ").title()


def predict(image_file, model, transform, label_encoder, device):
    """Run inference and return nicely formatted outputs for display."""
    tmp_path = "temp_upload.jpg"
    with open(tmp_path, "wb") as f:
        f.write(image_file.getvalue())

    # Predict
    class_name, confidence, all_probs = predict_image(
        model, tmp_path, transform, device, label_encoder
    )

    # Top-5
    class_indices = np.argsort(all_probs)[::-1][:5]
    top_classes = [label_encoder.inverse_transform([idx])[0] for idx in class_indices]
    formatted_top_classes = [format_class_name(x) for x in top_classes]
    formatted_primary_class = format_class_name(class_name)
    top_probabilities = [float(all_probs[idx]) * 100 for idx in class_indices]

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return (
        class_name,
        formatted_primary_class,
        float(confidence),
        top_classes,
        formatted_top_classes,
        top_probabilities,
    )


def display_prediction(original_class, formatted_class, confidence, top_classes, formatted_top_classes, top_probabilities):
    st.markdown(f"<h2 class='prediction-header'>Diagnosis: {formatted_class}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='confidence-text'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    prediction_df = pd.DataFrame({"Disease": formatted_top_classes, "Confidence": top_probabilities})

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(prediction_df["Disease"], prediction_df["Confidence"])
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Disease")
    ax.set_title("Top 5 Predictions")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{prediction_df['Confidence'][i]:.2f}%", va="center")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h3 class='sub-header'>Detailed Predictions</h3>", unsafe_allow_html=True)
    styled_df = prediction_df.style.format({"Confidence": "{:.2f}%"})
    st.table(styled_df)


def display_disease_info(class_name: str):
    """Show a small knowledge panel for a subset of classes (extend as needed)."""
    disease_info = {
        "Tomato_healthy": {
            "description": "Healthy tomato plants show vibrant green leaves, strong stems, and normal fruit development.",
            "causes": "Proper care, adequate watering, good sunlight exposure, and regular fertilization.",
            "treatment": "Continue regular care practices to maintain plant health.",
            "prevention": "Regular monitoring, balanced nutrition, appropriate watering, and good air circulation.",
        },
        "Potato___Late_blight": {
            "description": "Late blight causes dark, water-soaked lesions on leaves and tubers, leading to rapid decay.",
            "causes": "Caused by Phytophthora infestans, favored by cool, moist conditions.",
            "treatment": "Use fungicides like metalaxyl and promptly remove infected plants.",
            "prevention": "Plant resistant varieties, ensure good drainage, and avoid overhead watering.",
        },
    }
    if class_name in disease_info:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Disease Information</h3>", unsafe_allow_html=True)
        info = disease_info[class_name]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Description**")
            st.write(info["description"])
            st.markdown("**Causes**")
            st.write(info["causes"])
        with col_b:
            st.markdown("**Treatment**")
            st.write(info["treatment"])
            st.markdown("**Prevention**")
            st.write(info["prevention"])


# --------- Main app ---------
def main():
    model, transform, label_encoder, class_names, device, config = load_model_resources()

    st.title("Plant Disease Classifier")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

        with st.expander("Available Plant Diseases for Classification"):
            formatted_classes = [format_class_name(name) for name in class_names]
            classes_df = pd.DataFrame({"Available Diseases": formatted_classes})
            st.table(classes_df)

    if uploaded_file:
        # Preview
        with col2:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception:
                st.warning("Could not preview the image, but we will still try to run prediction.")

        (
            class_name, formatted_class, confidence,
            top_classes, formatted_top_classes, top_probabilities
        ) = predict(uploaded_file, model, transform, label_encoder, device)

        display_prediction(class_name, formatted_class, confidence, top_classes, formatted_top_classes, top_probabilities)
        display_disease_info(class_name)
    else:
        with col2:
            st.info("Please upload an image to see the preview, predictions and details.")

    # Sidebar info (avoid undefined variables)
    with st.sidebar:
        st.markdown("<h2 style='color: #2E7D32;'>About the Model</h2>", unsafe_allow_html=True)
        st.write("This application uses a CNN to classify plant diseases from leaf images.")
        st.write(f"**Number of Classes:** {len(load_model_resources()[3])}")
        st.markdown("---")
        st.caption("© 2025 Plant Disease Classifier - All Rights Reserved")


if __name__ == "__main__":
    main()
