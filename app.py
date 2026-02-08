import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️‍♂️")

# 2. Load Model (Cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    # Make sure 'frequency_detector_v2.keras' is in the same folder!
    return tf.keras.models.load_model('frequency_detector_v2.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Did you download the .keras file?")
    st.stop()

# 3. Frequency Conversion Function (Same as Kaggle)
def apply_fft(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)
    return np.expand_dims(magnitude_spectrum, axis=-1)

# 4. The Website UI
st.title("🕵️‍♂️ Frequency-Domain Deepfake Detector")
st.write("Upload an image to detect if it's **Real** or **AI-Generated** using Frequency Analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Analyzing Frequency Spectrum..."):
            
            # --- THE FIX: SMART CROP INSTEAD OF RESIZE ---
            # 1. Convert to numpy array
            img_array = np.array(image)
            height, width, _ = img_array.shape
            
            # 2. Calculate the center
            center_y, center_x = height // 2, width // 2
            
            # 3. Crop a 32x32 patch from the center
            # (This preserves the original pixel quality/artifacts!)
            start_y = center_y - 16
            start_x = center_x - 16
            img_cropped = img_array[start_y:start_y+32, start_x:start_x+32]
            
            # 4. Display what the model actually sees
            st.write("### 🔍 Analyzing this specific patch:")
            st.image(img_cropped, caption="Center 32x32 Crop (Uncompressed)", width=100)
            
            # 5. Transform & Predict
            freq_input = apply_fft(img_cropped)
            freq_input = np.expand_dims(freq_input, axis=0)
            
            prediction = model.predict(freq_input)
            confidence = prediction[0][0]
            
            # --- END OF FIX ---
            
            # Result Logic (Same as before)
            if confidence > 0.5:
                st.error(f"🚨 **FAKE (AI Generated)**")
                st.progress(int(confidence * 100))
                st.write(f"Confidence: {confidence:.2%}")
            else:
                st.success(f"✅ **REAL (Human)**")
                st.progress(int((1 - confidence) * 100))
                st.write(f"Confidence: {(1 - confidence):.2%}")

# Add this at the very end of app.py
with st.sidebar:
    st.header("📝 User Guide")
    st.markdown("""
    **How it works:**
    This AI doesn't look for "weird eyes." It looks for **invisible mathematical artifacts** in the frequency domain.
    
    **Best Results:**
    - Works best on **GAN-generated faces** (e.g., *ThisPersonDoesNotExist.com*).
    - May struggle with highly smoothed **Diffusion images** (e.g., DALL-E 3).
    
    **Why a crop?**
    We analyze a specific 32x32 patch to preserve the original pixel artifacts, rather than shrinking the whole image.
    """)