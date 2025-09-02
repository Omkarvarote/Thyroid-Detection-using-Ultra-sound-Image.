# 🧠 Thyroid Nodule Classification using Ultrasound (CNN + Streamlit)

This project is a **deep learning-based web app** that classifies thyroid nodules as **Benign** or **Malignant** using ultrasound images.  
The model is built with **TensorFlow/Keras** and deployed using **Streamlit**.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://thyroid-detection-using-ultrasound-axvaqx8bxkfa72yjpxnj7h.streamlit.app/)


## ⚙️ Installation & Usage (Run Locally)

1. **Clone the repository**  
```bash
git clone https://github.com/OmBhutkar/Thyroid-detection-using-ultrasound.git
cd Thyroid-detection-using-ultrasound
````

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run streamlit_app.py
```

---

## 🖼️ How It Works

1. Upload a thyroid **ultrasound image** (`.jpg`, `.jpeg`, `.png`).
2. The image is preprocessed and passed into the trained **CNN model**.
3. The model predicts whether the nodule is **Benign** or **Malignant**.
4. The app displays the result with confidence score.

---

## 📊 Model Details

* **Architecture:** Convolutional Neural Network (CNN)
* **Framework:** TensorFlow / Keras
* **Input Size:** 128×128
* **Output Classes:**

  * `Benign`
  * `Malignant`
  * `Normal Thyroid`

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **TensorFlow / Keras**
* **Streamlit**
* **scikit-learn**
* **Pillow**
* **NumPy**

---

## 🙌 Acknowledgements

* Dataset: Thyroid ultrasound images (public medical imaging datasets).
* Frameworks: TensorFlow, Streamlit, scikit-learn.

---

## 👨‍💻 Author

**Om Bhutkar**
📌 [GitHub Profile](https://github.com/OmBhutkar)

