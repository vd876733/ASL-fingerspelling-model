# ASL Fingerspelling Recognition — Local Integration

This repository contains a static frontend in `static/` and a simple Flask backend (`app.py`) that can accept base64 frames from the frontend and return a predicted letter and confidence. The backend attempts to use the existing `Model/keras_model.h5` and `Model/labels.txt` if present, otherwise it returns mock responses.

Quick start (Windows PowerShell):

1. Create a virtual environment and activate it (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r static\requirements.txt
```

3. Run the backend from the `static` folder so it serves the frontend files:

```powershell
cd static
python app.py
```

4. Open a browser and go to http://localhost:5000/ to use the frontend. The frontend will stream your webcam, capture frames, and POST them to `/predict`.

Notes:
- Ensure your model files are placed in `static/Model/keras_model.h5` and `static/Model/labels.txt` for live predictions.
- On Windows you may need to allow camera and localhost access in your browser.
