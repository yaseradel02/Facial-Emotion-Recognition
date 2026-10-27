
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Facial Emotion Recognition",
    layout="centered"
)

st.title("Facial Emotion Recognition")
st.write("Upload an image or use your webcam to predict facial emotions.")

# Model path (change if needed)
MODEL_PATH = Path(__file__).parent / "emotion_model.keras"

# Load model
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, compile=False)

if not MODEL_PATH.exists():
    st.warning("Model file not found. Please check the model path.")
    st.stop()

model = get_model()

# Class names (must match training order)
class_names = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

# Image input method
method = st.radio(
    "Choose input method:",
    ["Upload Image", "Use Webcam"],
    horizontal=True
)

image_file = None

if method == "Upload Image":
    image_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take a picture")

if image_file is not None:

    image = Image.open(image_file).convert("RGB")

    st.image(image, caption="Selected Image", width=280)

    if st.button("Predict Emotion", type="primary"):

        # Preprocess image
        img = image.convert("L")
        img = img.resize((48, 48))
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.reshape(1, 48, 48, 1)

        # Predict probabilities
        predictions = model.predict(img, verbose=0)[0]

        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        st.success(
            f"Predicted Emotion: {class_names[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        # Show all emotion probabilities
        st.subheader("Emotion Probabilities")

        probabilities = pd.DataFrame({
            "Emotion": class_names,
            "Probability": predictions * 100
        })

        probabilities = probabilities.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probabilities.set_index("Emotion")
        )

        st.dataframe(
            probabilities.style.format({
                "Probability": "{:.2f}%"
            }),
            hide_index=True,
            use_container_width=True
        )
