# ClipTool: an automated video clipping engine

**Turn a long podcast or talk into short 45–60 second clips with burned-in subtitles.** ClipTool transcribes the video with WhisperX, asks Gemini to split the transcript into topics and score each one for engagement, then cuts the best non-overlapping segments with FFmpeg.

> Personal project, 2025 · Python · WhisperX, PySceneDetect, OpenCLIP, Google Gemini, FFmpeg

## The problem

Short-form platforms reward 45–60 second clips, but finding the strongest minute in a one-hour recording is slow manual work. You have to watch the whole thing, judge which moments stand on their own, trim them to length and add captions. ClipTool automates the search and the cutting. It uses an LLM to judge which topics are worth clipping, and word-level timestamps to place the cuts and subtitles.

## What it does

| | |
|---|---|
| **Word-level transcript** | WhisperX transcribes the extracted audio with per-word timestamps. Speaker labels are used when present, otherwise every word is tagged `SPEAKER_00`. |
| **Scene and frame features** | PySceneDetect finds scene cuts and OpenCLIP (`ViT-L-14`, `laion2b_s32b_b82k`) embeds sampled frames. |
| **Pass 1: semantic chunking** | Gemini reads the line-numbered transcript and returns topic chunks (`title`, `start_line`, `end_line`). |
| **Pass 2: engagement scoring** | Each chunk is scored by Gemini on a 6-part rubric with a rationale per score, plus an `overall_score`. |
| **Boundary refinement** | Highest-scoring chunks are mapped back to word timestamps, padded by 0.5 s, fitted to 45–60 s and kept only if they don't overlap an already chosen clip. |
| **Clipping and packaging** | FFmpeg cuts each clip, an `.srt` is built from the chunk's word timings, subtitles are burned in, and a per-video `_analysis.json` records times, scores and titles. |
| **Resumable stages** | `--stage step3` re-runs the LLM analysis from a saved transcript. `--stage final` re-cuts clips from saved candidates without calling the LLM. |

The engagement rubric (from the score keys in [ClipperTool/output/](ClipperTool/output/)):

| Dimension | Key |
|---|---|
| Narrative cohesion | `narrative_cohesion` |
| Informational density | `informational_density` |
| Emotional impact | `emotional_impact` |
| Novelty and hook quality | `novelty_hook_quality` |
| Clarity and actionability | `clarity_actionability` |
| Social currency | `social_currency` |

## How it works

```
input video
  │
  ├─ [1/5] extract_audio            → temp/<name>.wav
  ├─ [2/5] run_whisperx             → temp/<name>_transcript.json (word timestamps)
  │        detect_scenes            → scene boundaries (PySceneDetect, threshold 27.0)
  │        get_frame_embeddings     → OpenCLIP frame embeddings
  ├─ [3/5] prompt_llm_segment       → temp/<name>_chunks.json       (Gemini pass 1: topics)
  │        prompt_llm_score × chunk → temp/<name>_candidates.json   (Gemini pass 2: 6-part rubric)
  ├─ [4/5] _generate_final_clips_metadata
  │        sort by overall_score → top MAX_FINAL_CLIPS → word boundaries ± 0.5 s
  │        → fit to 45–60 s → drop overlaps → temp/<name>_final_clips_metadata.json
  └─ [5/5] cut_video_segment_precise → create_subtitle_file → burn_subtitles_onto_video
           → output/<name>_clip_NN_with_subs.mp4 + output/<name>_analysis.json
```

**Notable design decisions:**
- **Two LLM passes instead of one.** Pass 1 only segments the transcript; pass 2 scores one chunk at a time. Each scoring prompt stays small, and the chunk boundaries are reused by later stages.
- **Line-numbered transcript.** The transcript is sent as one word per numbered line, so Gemini returns line numbers rather than timestamps. Code then maps those lines back to WhisperX word times.
- **Subtitles from exact word boundaries.** The video cut gets a 0.5 s buffer for natural flow, but the stored `first_word_time` / `last_word_time` are the unpadded word times used for the subtitles.
- **Every stage writes its output to `temp/`.** Transcription is the slowest step, so the pipeline can restart from the LLM stage or the clipping stage without redoing it.

## Results

The repo includes the summary files from two runs in [ClipperTool/output/](ClipperTool/output/). No source videos or clips are committed.

| Run | Clips | Lengths | Overall scores | Example title |
|---|---|---|---|---|
| [my_podcast_analysis.json](ClipperTool/output/my_podcast_analysis.json) | 4 | 60 s each | 8.3, 8.2, 7.8, 7.8 | "Finding Beauty in Unexpected Places" |
| [test_analysis.json](ClipperTool/output/test_analysis.json) | 4 | 60, 45, 60, 60 s | 6.2, 5.2, 4.7, 4.5 | "Weekly Activities: Spending and Travel Plans" |

Each entry carries all six rubric scores with Gemini's written rationale. The two files have slightly different schemas. `test_analysis.json` matches the current `clipping_pipeline.py` output, with separate video and speech start/end times. `my_podcast_analysis.json` uses the older `start_time` / `end_time` format.

## Repository layout

```
ClipperTool/
├── main.py                      CLI entry point: --stage full | step3 | final, --temp-dir
├── config.py                    API keys from .env, model names, clip length, thresholds, paths
├── requirements.txt             torch, whisperx (git), pyannote.audio 3.3.2, scenedetect, open-clip-torch, google-generativeai, …
├── pipelines/
│   ├── clipping_pipeline.py     VideoClippingPipeline: the 5-stage pipeline used by main.py
│   ├── resume_pipeline.py       older standalone "resume from step 3" pipeline (not used by main.py)
│   └── final_clipping.py        older standalone stage-5 pipeline (not used by main.py, see Limitations)
└── output/
    ├── my_podcast_analysis.json sample run summary
    └── test_analysis.json       sample run summary
```

## Quick start

**Prerequisites:**
- Python 3.9+.
- FFmpeg on your `PATH`.
- A CUDA GPU is recommended. `config.py` falls back to CPU automatically.

```bash
git clone https://github.com/LoheshM/ClipTool.git
cd ClipTool/ClipperTool
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `ClipperTool/.env` (there is no `.env.example` in the repo):

| Variable | Needed? | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini key from [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `HUGGINGFACE_HUB_TOKEN` | For diarization | Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept the terms for `pyannote/speaker-diarization-3.1`. |

Run from inside `ClipperTool/`:

```bash
python main.py path/to/video.mp4                 # full pipeline (stages 1–5)
python main.py path/to/video.mp4 --stage step3   # resume from LLM analysis, needs temp/<name>_transcript.json
python main.py path/to/video.mp4 --stage final   # clipping only, needs temp/<name>_transcript.json and _candidates.json
```

Settings in [config.py](ClipperTool/config.py):

| Setting | Value in repo | Meaning |
|---|---|---|
| `LLM_MODEL_NAME` | `gemini-1.5-flash-latest` | Gemini model for both passes |
| `WHISPERX_MODEL` / `WHISPERX_COMPUTE_TYPE` | `medium` / `int8` | Main transcription model |
| `WHISPERX_CLIP_MODEL` | `base` | Smaller model intended for per-clip re-transcription |
| `ENABLE_SPEAKER_DIARIZATION` | `False` | pyannote diarization switch |
| `TARGET_CLIP_DURATION` | `(45, 60)` | Min and max clip length in seconds |
| `MAX_FINAL_CLIPS` | `4` | Top-scored candidates considered per video |
| `SCENE_DETECTION_THRESHOLD` | `27.0` | PySceneDetect threshold |
| `FRAME_SAMPLE_RATE` | `1` (`2` on Windows) | Frame sampling for OpenCLIP embeddings |
| `CLIP_MODEL_NAME` / `CLIP_PRETRAINED_DATASET` | `ViT-L-14` / `laion2b_s32b_b82k` | OpenCLIP weights |
| `TEMPDIR` / `OUTPUT_DIR` | `./temp` / `./output` | Intermediate and final files |

**Troubleshooting** (from the original README):
- *FFmpeg not found:* make sure FFmpeg is installed and on `PATH`.
- *CUDA out of memory:* reduce the WhisperX batch size, or use a smaller `WHISPERX_MODEL`.
- *API key errors:* check `.env`.

## Limitations

- **The `utils/` package is not in the repo.** All three pipelines import `utils.video_utils`, `utils.llm_utils` and `utils.ffmpeg_utils`, which were never committed. The code will not run until those modules are added back. This includes the Gemini prompts and the FFmpeg/WhisperX wrappers.
- **Hard-coded FFmpeg path.** `main.py` prepends `C:\Users\Lohesh\Documents\Packages\ffmpeg\bin` to `PATH`. Remove or change that line on other machines.
- **Visual features are computed but barely used.** Scene cuts and OpenCLIP embeddings are extracted in stage 2, but the scoring prompt only receives the string "Visual context analysis completed". `--stage step3` skips visual analysis entirely.
- **`final_clipping.py` is broken.** Its stage-5 loop is indented at class level and calls functions that are not imported. `main.py` does not use it; it uses `VideoClippingPipeline.run_final_clipping` instead.
- **Clip lengths are approximate.** Long chunks are trimmed evenly from both ends, which can cut mid-sentence. Short chunks are padded with surrounding footage.
- **Model names may be outdated.** `gemini-1.5-flash-latest` and the `google-generativeai` SDK date from 2025.

**Ideas noted for future work** (from the original README): vertical auto-reframing, animated captions, performance analytics, fine-tuned models, a Gradio web UI.

## Credits

Built on [WhisperX](https://github.com/m-bain/whisperx), pyannote.audio, PySceneDetect, OpenCLIP, Google Gemini and FFmpeg.

## Licence

MIT. See [LICENSE](LICENSE). The third-party models and libraries above keep their own licences and terms. In particular, the pyannote diarization model requires accepting its terms on Hugging Face.
