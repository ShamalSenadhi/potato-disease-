# Potato Leaf Disease Classifier — Streamlit Dashboard

A Streamlit web app that uses a trained CNN (TensorFlow/Keras) to classify potato
leaf images into 7 classes: Bacteria, Fungi, Healthy, Nematode, Pest, Phytophthora, Virus.

## Project structure

```
potato-disease-app/
├── app.py               # Streamlit app
├── model_v1.keras        # Your trained model (add this yourself)
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## 1. Add your trained model

Copy your saved model file (e.g. `model_v1.keras` or `model_v1.h5`, from the Colab
notebook where you ran `model.save(...)`) into this folder, next to `app.py`.

If your file has a different name, update `MODEL_PATH` at the top of `app.py`.

## 2. Run locally

```bash
# create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 3. Push to GitHub

```bash
# from inside the potato-disease-app folder
git init
git add .
git commit -m "Initial commit: potato leaf disease Streamlit app"

# create a new empty repo on GitHub first, then:
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

> ⚠️ Model file size: GitHub blocks files over 100MB via normal `git push`.
> If your `.keras`/`.h5` file is large, use **Git LFS**:
> ```bash
> git lfs install
> git lfs track "*.keras" "*.h5"
> git add .gitattributes
> git add model_v1.keras
> git commit -m "Add model with Git LFS"
> git push
> ```

## 4. Deploy for free (Streamlit Community Cloud)

1. Push your code to GitHub (steps above).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, select your repo, branch (`main`), and set the main file
   path to `app.py`.
4. Click **Deploy**. Streamlit installs everything from `requirements.txt`
   automatically and gives you a public URL.

## Updating the app later

```bash
git add .
git commit -m "Update model / app"
git push
```

Streamlit Community Cloud auto-redeploys on every push to the connected branch.
