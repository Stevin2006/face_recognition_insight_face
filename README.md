#  Real-Time Face Recognition System

A lightweight, real-time face recognition system built with **InsightFace (buffalo_l)**, **ChromaDB**, and **OpenCV**. Detects and identifies faces from a webcam stream using **RetinaFace** for detection and **ArcFace** for 512-dimensional embedding extraction. Inference runs on a dedicated background thread so the camera feed stays smooth with zero freezing — even on CPU-only machines.

Register any number of users via webcam, store their face embeddings in a persistent ChromaDB vector database, and recognize them in real time using cosine similarity search.

---

## 📸 How It Works

```
Webcam Frame
     │
     ▼
RetinaFace (Detection)
     │
     ▼
ArcFace (512-D Embedding)        ← Runs in background thread
     │
     ▼
ChromaDB (Cosine Similarity Search)
     │
     ▼
Name + Distance Score overlaid on live feed
```

---

## 🧠 Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Face Detection   | RetinaFace (via InsightFace)        |
| Face Embedding   | ArcFace 512-D (via InsightFace)     |
| Vector Database  | ChromaDB (cosine similarity)        |
| Camera & Display | OpenCV                              |
| Inference        | ONNX Runtime (CPU)                  |
| Language         | Python 3.12                         |

---

## 📁 Project Structure

```
face-recognition/
│
├── register.py        # Register a new user via webcam
├── recognize.py       # Real-time face recognition
├── requirements.txt   # Python dependencies
├── README.md          # This file
│
└── face_db/           # Auto-created by ChromaDB on first run
```

---

## ⚙️ System Requirements

- Python **3.12** (Windows 64-bit)
- Webcam (USB or built-in)
- No GPU required — runs fully on CPU
- Internet connection (first run only — to download the `buffalo_l` model ~300 MB)

---

## 🚀 Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/face-recognition.git
cd face-recognition
```

### Step 2 — Create a virtual environment

```bash
python -m venv face
face\Scripts\activate
```

> On Mac/Linux: `source face/bin/activate`

### Step 3 — Install InsightFace (pre-built wheel)

InsightFace requires C++ build tools to compile from source on Windows. Use this pre-built wheel instead — no compiler needed:

```bash
pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl
```

> **For other Python versions**, replace `cp312-cp312` with your version:
> - Python 3.10 → `cp310-cp310`
> - Python 3.11 → `cp311-cp311`

### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Important:** Always install the InsightFace wheel **before** running `pip install -r requirements.txt`. This ensures `numpy<2.0` is respected and prevents binary incompatibility errors.

### Step 5 — Verify installation

```bash
python -c "import insightface; print('InsightFace OK:', insightface.__version__)"
python -c "import chromadb; print('ChromaDB OK')"
python -c "import cv2; print('OpenCV OK:', cv2.__version__)"
```

Expected output:
```
InsightFace OK: 0.7.3
ChromaDB OK
OpenCV OK: 4.x.x
```

---

## 🎬 Usage

### 1. Register a User

```bash
python register.py
```

- Enter the person's name when prompted
- The webcam opens — look at the camera
- The script captures **7 face samples** automatically
- Embeddings are saved to ChromaDB in `./face_db/`
- Repeat for each person you want to register

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit registration early |

**Example output:**
```
Enter the name to register: Alice

[INFO] Registering: 'Alice'  |  Samples needed: 7
[INFO] Loading InsightFace model …
[INFO] Model loaded.
[INFO] Inference thread started. Webcam is live …

  ✔  Captured 1/7
  ✔  Captured 2/7
  ✔  Captured 3/7
  ✔  Captured 4/7
  ✔  Captured 5/7
  ✔  Captured 6/7
  ✔  Captured 7/7

[INFO] Saving 7 embeddings to ChromaDB …
[SUCCESS] 'Alice' registered with 7 samples.
[INFO] Total embeddings in DB: 7
```

---

### 2. Run Real-Time Recognition

```bash
python recognize.py
```

- Webcam opens immediately at full FPS
- Detected faces are boxed and labelled with name + distance score
- **Green box** = recognized user
- **Red box** = unknown person

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit recognition |

**Example output:**
```
[INFO] Loading InsightFace model …
[INFO] Model ready.
[INFO] DB: 14 embeddings  |  Threshold: 0.45
[INFO] Inference thread started. Press 'q' to quit.
```

---

## 🎛️ Configuration

All settings are at the top of each script. Key values you may want to tune:

### `register.py`

| Variable          | Default | Description                                      |
|-------------------|---------|--------------------------------------------------|
| `SAMPLES_REQUIRED`| `7`     | Number of face samples to capture per user       |
| `CAPTURE_INTERVAL`| `0.8`   | Seconds between captures (avoid duplicate frames)|
| `MIN_FACE_SIZE`   | `60`    | Minimum face width/height in pixels              |
| `DET_SIZE`        | `(320, 320)` | Detection resolution — lower = faster       |

### `recognize.py`

| Variable                 | Default | Description                                    |
|--------------------------|---------|------------------------------------------------|
| `RECOGNITION_THRESHOLD`  | `0.45`  | Cosine distance cutoff for known/unknown       |
| `MIN_FACE_SIZE`          | `60`    | Minimum face width/height in pixels            |
| `DET_SIZE`               | `(320, 320)` | Detection resolution — lower = faster     |
| `SMOOTH_WINDOW`          | `5`     | Frames used for label smoothing                |

### Tuning the threshold

```
0.0 ──────────────────────┬──────────────────── 1.0
      very strict        0.45             very lenient
   (fewer false matches)            (more false matches)
```

- Lower (`0.35`) → stricter, fewer false positives, may miss valid faces
- Higher (`0.55`) → more lenient, better in varied lighting/angles

---

## 🧵 Threading Architecture

InsightFace inference is offloaded to a background thread so the camera display never blocks:

```
Main Thread                      Inference Thread
────────────────────             ──────────────────────────────
cap.read() → frame               InsightFace detect()
      │                                 │
      ▼                                 ▼
input_queue.put()   ──────────►  input_queue.get()
                                        │
display runs freely              ArcFace embed()
at full FPS using                       │
cached results                   ChromaDB query()
      │                                 │
result_queue.get()  ◄──────────  result_queue.put()
      │
draw bounding boxes
```

- **`maxsize=1` queues** — always processes the latest frame, never a backlog
- **`daemon=True`** — inference thread exits automatically when main thread closes
- **Non-blocking submit** — camera loop never stalls waiting for AI

---

## 🐛 Troubleshooting

### ❌ `numpy.dtype size changed` error
```bash
pip install "numpy<2.0"
```
Then reinstall InsightFace wheel.

---

### ❌ `Failed building wheel for insightface`
You are installing from source. Use the pre-built wheel instead:
```bash
pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl
```

---

### ❌ Camera is frozen / stuttering
- Make sure you are using the threaded version of the scripts (this repo)
- Check no other application is using the webcam
- Try changing the camera index in `cv2.VideoCapture(0, ...)` from `0` to `1`

---

### ❌ `Collection 'faces' not found`
You need to register at least one user before running `recognize.py`:
```bash
python register.py
```

---

### ❌ Model download fails on first run
The `buffalo_l` model (~300 MB) is downloaded automatically on first run. Make sure you have an active internet connection. The model is cached locally after the first download at:
```
C:\Users\<you>\.insightface\models\buffalo_l\
```

---

### ❌ `Cannot open webcam`
- Check your webcam is connected and not in use by another app
- Try `cv2.VideoCapture(1, cv2.CAP_DSHOW)` instead of `0`

---

## 📦 Dependencies

| Package          | Version     | Purpose                              |
|------------------|-------------|--------------------------------------|
| `insightface`    | `0.7.3`     | RetinaFace detection + ArcFace embed |
| `onnxruntime`    | `>=1.16.0`  | ONNX model inference on CPU          |
| `opencv-python`  | `>=4.8.0`   | Webcam capture and display           |
| `chromadb`       | `>=0.4.0`   | Vector DB for embedding storage      |
| `numpy`          | `<2.0`      | Numerical operations                 |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface) — ArcFace + RetinaFace models
- [ChromaDB](https://github.com/chroma-core/chroma) — Vector database
- [Gourieff](https://github.com/Gourieff) — Pre-built Windows wheels for InsightFace
