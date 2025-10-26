import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load trained model
model = load_model("mnist_cnn_model.h5")

st.title("🧠 Handwritten Digit Recognition App")
st.write("Draw a digit (0–9) below and let the CNN predict it!")

# Create a drawing canvas using streamlit-drawable-canvas
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=200,
    height=200,
    key="canvas",
)

if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
    # Convert RGBA → Grayscale
        img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
    
    # Invert so black digit on white background
        img = ImageOps.invert(img)
    
    # Crop to remove extra white space
        img = img.crop(img.getbbox())

    # Resize and center to 28x28 like MNIST
        img = img.resize((28, 28))
    
    # Convert to array and normalize
        img_arr = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0

    # Predict
        pred = model.predict(img_arr)
        st.subheader(f"🧾 Prediction: {np.argmax(pred)}")

    else:
        st.warning("Please draw a digit first!")
