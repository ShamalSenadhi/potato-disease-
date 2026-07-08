# 🥔 Potato Leaf Disease Detector

A deep learning-powered web dashboard that identifies diseases in potato leaves from a single photo and provides actionable treatment and prevention guidance — built with TensorFlow/Keras and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

Potato crops are highly vulnerable to a range of diseases — bacterial, fungal, viral, nematode, and pest-related — that can significantly reduce yield if not identified and treated early. Manual diagnosis by agricultural experts is often slow, subjective, and inaccessible to smallholder farmers.

This project addresses that gap with a Convolutional Neural Network (CNN) trained to classify potato leaf images into **7 categories**, paired with an interactive Streamlit dashboard that goes beyond raw prediction — offering symptom explanations, causes, treatment recommendations, and prevention strategies for each detected condition.

## ✨ Features

- 🔍 **Instant disease detection** — upload a leaf photo and get a prediction with confidence score in seconds
- 📊 **Confidence breakdown** — visual chart showing the model's confidence across all 7 classes
- 💊 **Treatment guidance** — specific, actionable recommendations for each detected disease
- 🛡️ **Prevention tips** — practical steps to avoid future outbreaks
- 📚 **Disease library** — a full reference guide covering symptoms, causes, treatment, and prevention for all classes, browsable without uploading an image
- 📈 **Session history** — tracks all predictions made in a session, with summary metrics and CSV export
- ⚠️ **Severity indicators** — flags high-risk diseases (e.g. Late Blight) that require urgent action

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | Convolutional Neural Network (CNN) |
| Framework | TensorFlow / Keras |
| Input size | 256 × 256 × 3 (RGB) |
| Classes | 7 — Bacteria, Fungi, Healthy, Nematode, Pest, Phytophthora, Virus |
| Training data | 3,076 labelled potato leaf images |
| Output | Softmax probability distribution over 7 classes |

The model was trained on an image dataset organised by class folders, using data augmentation (random flip, rotation, zoom) to improve generalisation, and evaluated on a held-out test split.

## 🗂️ Project Structure

```
potato-disease-app/
├── app.py                # Streamlit dashboard (main application)
├── model_v1.keras        # Trained CNN model (not included — see below)
├── requirements.txt      # Python dependencies
├── packages.txt           # System-level dependencies (for Streamlit Cloud)
├── runtime.txt            # Python version pin
├── Dockerfile              # Container config (for Hugging Face Spaces / Docker deployment)
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 (recommended — TensorFlow does not yet support newer versions)
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Add the trained model

Place your trained model file (`model_v1.keras`) in the project root, next to `app.py`. If your file has a different name or version, update `MODEL_PATH` at the top of `app.py` accordingly.

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

## ☁️ Deployment

### Option A — Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select the repo/branch, and set the main file to `app.py`.
4. Under **Advanced settings**, set the **Python version** to `3.11` (TensorFlow wheels are not yet available for newer Python versions).
5. Deploy.

> ⚠️ If your model file exceeds GitHub's 100MB limit, use [Git LFS](https://git-lfs.github.com/):
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes model_v1.keras
> git commit -m "Add model with Git LFS"
> git push
> ```

### Option B — Docker / Hugging Face Spaces

A `Dockerfile` is included, pinned to Python 3.11, for platforms that support container-based deployment:

```bash
docker build -t potato-disease-app .
docker run -p 8501:8501 potato-disease-app
```

To deploy on Hugging Face Spaces, create a new Space with **SDK: Docker**, then push this repo to it.

## 🖥️ Usage

1. Navigate to the **Detect Disease** page.
2. Upload a clear, well-lit photo of a single potato leaf.
3. View the predicted class, confidence score, and confidence breakdown across all categories.
4. Review the **Symptoms & Causes**, **Treatment**, and **Prevention** tabs for guidance specific to the detection.
5. Browse the **Disease Library** at any time for reference information on all 7 classes.
6. Check **Session History** to review or export past predictions made during your session.

## 🛠️ Tech Stack

- **Model:** TensorFlow, Keras
- **Dashboard:** Streamlit
- **Data handling:** NumPy, Pandas
- **Image processing:** Pillow

## ⚠️ Disclaimer

This tool is intended for **preliminary screening and educational purposes only**. It is not a substitute for professional agricultural diagnosis. Always consult a qualified agronomist or plant pathologist to confirm disease identification and determine an appropriate treatment plan before taking action on a crop.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Developed as part of the **Advanced Machine Learning (COM 763)** module.
- Dataset: potato leaf disease images labelled across 7 classes (Bacteria, Fungi, Healthy, Nematode, Pest, Phytophthora, Virus).

## 📬 Contact

For questions or feedback, please open an issue on this repository.
