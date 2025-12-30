import os
import platform
from dotenv import load_dotenv
import torch

load_dotenv()  # Loads variables from a .env file

# --- API Keys and Tokens (CRITICAL - GET THESE) ---
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") # Required for pyannote.audio
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Required for Gemini

# --- Model Settings ---
WHISPERX_MODEL = "medium" # "medium", "large-v2", etc.
WHISPERX_COMPUTE_TYPE = "int8" # "float16" for GPU, "int8" for CPU
PYANNOTOTE_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTOTE_PIPELINE = "pyannote-voice-activity-detection"
ENABLE_SPEAKER_DIARIZATION = False 
# LLM Settings
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# Visual Analysis Settings
SCENE_DETECTION_THRESHOLD = 27.0 # For PySceneDetect (adjust based on content)
FRAME_SAMPLE_RATE = 1 # Sample 1 frame per second for embedding
CLIP_MODEL_NAME = "ViT-L-14" # OpenCLIP model
CLIP_PRETRAINED_DATASET = "laion2b_s32b_b82k" # OpenCLIP pretrained weights
# Add this to your config.py for clip processing
WHISPERX_CLIP_MODEL = "base"  # Use smaller model for clips for faster processing
WHISPERX_CLIP_COMPUTE_TYPE = "int8"  # Computation type for clip processing
# Clip Settings
TARGET_CLIP_DURATION = (45, 60) # Target clip length in seconds (min, max)
MAX_FINAL_CLIPS = 4 # Maximum number of clips to generate per video

# System Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPDIR = "./temp" # Directory for intermediate files
OUTPUT_DIR = "./output" # Directory for final clips

if platform.system() == "Windows":
    # Disable symlinks warning
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    # Use a more compatible approach
    FRAME_SAMPLE_RATE = 2  # Sample less frequently on Windows to save memory