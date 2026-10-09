# FrameCheck

FrameCheck is a Streamlit app that analyzes images and sampled video frames with the `Organika/sdxl-detector` model to estimate whether media is AI-generated or human-made. It also provides image metadata and EXIF inspection, SHA-256 hashing, and JPEG Error Level Analysis (ELA).

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app** and select that repository, branch, and `app.py` as the main file.
3. Deploy. Streamlit installs packages from `requirements.txt` and applies the optional theme in `.streamlit/config.toml`.

The app runs inference on the Streamlit server. Do not upload secrets or private data to the repository. Uploaded media is processed by the hosted app.

## Main files

- `app.py` — Streamlit interface and image/video analysis.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — Optional dark theme and server settings.
- `.gitignore` — Excludes local environments, caches, and secrets from Git.

## Model

The app loads `Organika/sdxl-detector` from Hugging Face through Transformers. The result is a model estimate and should not be treated as definitive proof of whether media is authentic.
