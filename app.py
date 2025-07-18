import streamlit as st
from model_helper import predict

st.set_page_config(page_title="Vehicle Damage Detection", page_icon="🚗", layout="centered")

st.markdown(
    """
    <style>
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #004aad;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .upload-area {
        border: 2px dashed #0078d7;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .upload-area:hover {
        border-color: #004aad;
    }
    .prediction-box {
        background-color: #e7f0fe;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #003366;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Vehicle Damage Detection</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drag and drop or click to upload an image file",
    type=["jpg", "png"],
    help="Upload an image of the vehicle to detect damage",
    label_visibility="visible",
)

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True, clamp=True)
    prediction = predict(image_path)

    st.markdown(f'<div class="prediction-box">Predicted Class: {prediction}</div>', unsafe_allow_html=True)
