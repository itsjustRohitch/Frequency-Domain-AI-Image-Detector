# 🕵️‍♂️ Frequency-Domain AI Image Detector
> **Catching AI by the invisible math it leaves behind.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📸 See It In Action

<img width="1918" height="946" alt="image" src="https://github.com/user-attachments/assets/c57c8f48-9edd-48ce-b8b0-bbc76646e6f6" />

> *"AI Images may look real to the eye, but their frequency spectrum tells a different story."*

---

## 🧐 What is this?
Most AI detectors look for "weird hands" or "mismatched eyes." **This project is different.** It uses **Frequency Domain Analysis** (Discrete Fourier Transform) to detect the invisible checkerboard artifacts left behind by Generative Adversarial Networks (GANs) during the upsampling process.

**Key Features:**
* **Invisible Detection:** Detects fakes even when they look perfect to humans.
* **Lightweight:** Uses a custom CNN trained on Frequency Spectrums, not heavy RGB pixels.
* **92% AUC Score:** High reliability on GAN-generated imagery.
* **Real-Time Web App:** Built with Streamlit for instant drag-and-drop testing.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Core Logic** | Python, NumPy, OpenCV (FFT) |
| **Model Training** | TensorFlow / Keras (CNN) |
| **Visualization** | Matplotlib, Seaborn |
| **Web Interface** | Streamlit |
| **Dataset** | [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |

---

## 🧠 The Science (Process Followed)

The pipeline transforms standard images into their **Frequency Spectrum** before feeding them into the Neural Network.

<img width="2076" height="1086" alt="image" src="https://github.com/user-attachments/assets/f10a195d-b5b8-408c-9893-ea6a9ef0821f" />

1. Preprocessing: Images are resized to 32x32 and converted to grayscale.
2. FFT Transformation: We apply a Fast Fourier Transform to reveal frequency distributions.
3. Training: A Convolutional Neural Network (CNN) learns to spot the specific "high-frequency grid" artifacts typical of GANs.
4. Inference: The model outputs a probability score (0.0 to 1.0).

---

## 📊 Performance Results
1. **Confusion Matrix:**
<img width="522" height="470" alt="image" src="https://github.com/user-attachments/assets/6943df1d-44fd-4245-b15b-21ce0584eade" />

2. **Classification Report:**

                precision    recall  f1-score   support
  
          Real       0.86      0.83      0.84     10000
          Fake       0.83      0.86      0.85     10000
  
      accuracy                           0.84     20000
     macro avg       0.84      0.84      0.84     20000
  weighted avg       0.84      0.84      0.84     20000

3. **ROC Curve:**
<img width="578" height="455" alt="image" src="https://github.com/user-attachments/assets/9fcf564a-4c2a-4aef-b71a-371345c98f57" />

---

## 🚀 How to Run Locally
1. **Clone the Repository**
`git clone [https://github.com/YOUR_USERNAME/frequency-domain-deepfake-detector.git](https://github.com/YOUR_USERNAME/frequency-domain-deepfake-detector.git)`
`cd frequency-domain-deepfake-detector`

2. **Create a Virtual Environment (Recommended)**
- Windows
`python -m venv venv`
`venv\Scripts\activate`

- Mac/Linux
`python3 -m venv venv`
`source venv/bin/activate`

4. **Install Dependencies**
`pip install -r requirements.txt`

5. **Run the App**
`streamlit run app.py`

---

## ⚠️ Limitations (The "Real Talk")

While this model is robust against GANs (StyleGAN, CycleGAN), it faces challenges with modern Diffusion Models (DALL-E 3, Midjourney).
- The Resolution Trap: Diffusion models use "denoising" which smooths out the high-frequency artifacts this model looks for.
- Compression: heavily compressed images (WhatsApp/Twitter) lose their frequency details, which can lower detection confidence.
- The Fix: Future versions will need a "Multi-Stream" network that combines standard RGB analysis with Frequency analysis to catch both types of fakes.

---

## 🔮 Future Advancements

If you use this project as a reference, here is how you can take it to the next level:
- Hybrid Model: Combine a pre-trained ResNet50 (for visual features) with this Frequency CNN.
- High-Res Input: Retrain the model on 256x256 inputs to catch finer artifacts in DALL-E 3 images.
- Video Support: Implement frame-by-frame analysis to detect Deepfake videos.

---

## 🤝 Contributing

- Found a bug? Want to improve the accuracy?
- Fork the repo.
- Create a new branch (git checkout -b feature-improvement).
- Commit your changes.
- Push to the branch and open a Pull Request.

---

## 📜 License
MIT License

Copyright (c) 2025 [ROHIT]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


