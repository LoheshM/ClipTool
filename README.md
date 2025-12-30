

# 🎬 Automated Video Clipping Engine

An end-to-end pipeline for automatically generating engaging **short-form clips (45–60 seconds)** from long-form videos using **multimodal AI analysis**.

---

## 🚀 Features

- **Automatic Clip Detection** – Identifies high-potential segments using engagement scoring
- **Multimodal Analysis** – Audio, visual, and semantic understanding combined
- **Perfect Subtitle Synchronization** – Frame-accurate subtitles using WhisperX
- **Open-Source Stack** – Built entirely with free, open-source technologies
- **Cost Optimized** – Two-pass LLM strategy minimizes API calls while maintaining quality

---

## 🏗️ Architecture

### 5-Stage Pipeline

1. **Ingestion & Pre-processing**
   - Extract audio
   - Prepare video streams

2. **Multimodal Feature Extraction**
   - **WhisperX** – Word-level transcription with speaker diarization
   - **PySceneDetect** – Visual scene boundary detection
   - **OpenCLIP** – Semantic frame embeddings

3. **Intelligence Core**
   - **Pass 1:** Semantic chunking to identify topics
   - **Pass 2:** Engagement scoring using a 6-dimensional rubric

4. **Boundary Refinement**
   - Algorithmic clip selection
   - Timing adjustments

5. **Clipping & Packaging**
   - FFmpeg-based cutting
   - Burned-in subtitles

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- FFmpeg installed system-wide
- CUDA-compatible GPU (recommended)

---

### Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/automated-video-clipper.git
cd automated-video-clipper

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg
# Ubuntu: sudo apt install ffmpeg
# Mac: brew install ffmpeg
# Windows: https://ffmpeg.org/

# 5. Configure API keys
cp .env.example .env


---

## 🔑 API Keys Required

### Google API Key

Used for **Gemini 1.5 Flash**

Get it from:
👉 [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

### Hugging Face Token

Used for **speaker diarization**

Get it from:
👉 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

⚠️ Accept terms for:

```
pyannote/speaker-diarization-3.1
```

---

## 📁 Project Structure

```text
automated-video-clipper/
├── main.py
├── config.py
├── requirements.txt
├── pipelines/
│   └── clipping_pipeline.py
├── utils/
│   ├── video_utils.py
│   ├── llm_utils.py
│   └── ffmpeg_utils.py
├── output/
└── temp/
```

---

## 🚦 Usage

### Basic Usage

```bash
# Run full pipeline
python main.py path/to/video.mp4 --stage full

# Resume from LLM analysis (Stage 3)
python main.py path/to/video.mp4 --stage step3

# Run only final clipping (Stage 5)
python main.py path/to/video.mp4 --stage final
```

---

### Stage Selection

* `--stage full` – Complete 5-stage pipeline (default)
* `--stage step3` – Resume from LLM analysis
* `--stage final` – Generate clips from existing analysis

---

## 📤 Output Structure

```text
output/
├── video_clip_01_with_subs.mp4
├── video_clip_01.srt
├── video_clip_02_with_subs.mp4
├── video_clip_02.srt
└── video_analysis.json
```

---

## 🔧 Configuration

Edit `config.py`:

```python
# LLM Settings
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# Clip Settings
TARGET_CLIP_DURATION = (45, 60)
MAX_FINAL_CLIPS = 5

# WhisperX Settings
WHISPERX_MODEL = "large-v2"
WHISPERX_CLIP_MODEL = "base"

# Visual Analysis
SCENE_DETECTION_THRESHOLD = 27.0
FRAME_SAMPLE_RATE = 1
```

---

## 🧠 Intelligence Core

### Engagement Scoring Rubric

Each segment is scored on:

1. Narrative Cohesion
2. Informational Density
3. Emotional Impact
4. Novelty & Hook Quality
5. Clarity & Actionability
6. Social Currency

---

### Multimodal Context Integration

* **Speaker Dynamics** – Conversations vs monologues
* **Visual Cues** – Scene changes, diagrams, screen sharing
* **Semantic Analysis** – CLIP-based frame understanding

---

## ⚡ Performance Tips

* Enable **CUDA** for 5–10× speedup
* Use `WHISPERX_MODEL = "medium"` for faster transcription
* Process videos sequentially
* Resume with `--stage step3` or `--stage final`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit changes

   ```bash
   git commit -m "Add AmazingFeature"
   ```
4. Push to branch

   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

---

## 📄 License

MIT License — see the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* **WhisperX** – Accurate timestamping
* **Google Gemini** – LLM analysis
* **FFmpeg** – Video processing
* **OpenCLIP** – Visual embeddings

---

## 🐛 Troubleshooting

### Common Issues

* **FFmpeg not found** – Ensure FFmpeg is installed and in PATH
* **CUDA out of memory** – Reduce WhisperX batch size
* **API key errors** – Check `.env` configuration
* **Subtitle sync issues** – Verify WhisperX clip model

---

## 📊 Future Enhancements

* Vertical video auto-reframing
* Dynamic animated captions
* Performance analytics
* Fine-tuned specialized models
* Web UI with Gradio

```

---

If you want, I can also:
- Compress this into a **short GitHub repo summary**
- Add **badges** (Python version, license, GPU, open-source)
- Optimize it for **GitHub SEO / discoverability**

Just say the word.
```
