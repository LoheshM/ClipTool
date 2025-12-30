import os
import json
from typing import List, Dict, Any, Tuple
from utils.llm_utils import initialize_llm, prompt_llm_segment, prompt_llm_score
from utils.ffmpeg_utils import cut_video_segment, create_subtitle_file
from config import *
# from utils.moviepy_utils import cut_video_segment_moviepy, extract_audio_moviepy

class ResumeClippingPipeline:
    def __init__(self):
        self.llm_model = initialize_llm(GOOGLE_API_KEY, LLM_MODEL_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run_from_step3(self, video_file_path: str, temp_dir: str = TEMPDIR):
        """Resumes pipeline from Step 3 (LLM Analysis) using saved intermediate files."""
        base_filename = os.path.splitext(os.path.basename(video_file_path))[0]
        print(f"[*] Resuming pipeline for: {base_filename} from Step 3")

        # Load previously saved intermediate files
        transcript_path = os.path.join(temp_dir, f"{base_filename}_transcript.json")
        chunk_path = os.path.join(temp_dir, f"{base_filename}_chunks.json")
        
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunks file not found: {chunk_path}")

        # --- STAGE 3: Intelligence Core ---
        print("[3/5] Analyzing content with LLM (resumed)...")
        
        # Load whisper result
        with open(transcript_path, 'r', encoding='utf-8') as f:
            whisper_result = json.load(f)
        
        # Load existing chunks or create new ones
        try:
            with open(chunk_path, 'r') as f:
                semantic_chunks = json.load(f)
            print("  Loaded existing semantic chunks")
        except:
            print("  No existing chunks found, generating new ones...")
            # Format transcript for LLM with line numbers
            lines = []
            for seg in whisper_result['segments']:
                for word in seg['words']:
                    line = f"SPEAKER_{word['speaker']}: {word['word']}"
                    lines.append(line)
            formatted_transcript = "\n".join([f"{i+1}: {line}" for i, line in enumerate(lines)])
            
            semantic_chunks = prompt_llm_segment(self.llm_model, formatted_transcript)
            with open(chunk_path, 'w') as f:
                json.dump(semantic_chunks, f, indent=2)

        # 3b. Pass 2: Engagement Scoring
        scored_candidates = []
        for chunk in semantic_chunks:
            start_line, end_line = chunk['start_line'], chunk['end_line']
            
            # Reconstruct chunk text from whisper result
            chunk_text_lines = []
            current_line = 1
            for seg in whisper_result['segments']:
                for word in seg['words']:
                    if start_line <= current_line <= end_line:
                        line_text = f"SPEAKER_{word['speaker']}: {word['word']}"
                        chunk_text_lines.append(f"{current_line}: {line_text}")
                    current_line += 1
            
            chunk_text = '\n'.join(chunk_text_lines)

            # Create simple context summaries
            speaker_list = list(set([line.split(':')[0] for line in chunk_text_lines if 'SPEAKER' in line]))
            speaker_ctx = f"Features speakers: {', '.join(speaker_list)}" if speaker_list else "Single speaker."
            visual_ctx = "Visual context: Standard analysis"  # Simplified for resume

            print(f"  Analyzing chunk: {chunk['title']}")
            score_result = prompt_llm_score(self.llm_model, chunk_text, speaker_ctx, visual_ctx)
            score_result['chunk_metadata'] = chunk
            scored_candidates.append(score_result)

        candidates_path = os.path.join(temp_dir, f"{base_filename}_candidates.json")
        with open(candidates_path, 'w') as f:
            json.dump(scored_candidates, f, indent=2)

        # --- Continue with STAGE 4 and 5 ---
        print("[4/5] Refining clip boundaries and selecting final clips...")
        # ... [Rest of your original Stage 4 and 5 code from clipping_pipeline.py] ...
        scored_candidates.sort(key=lambda x: x['overall_score'], reverse=True)

        final_clips_metadata = []
        selected_time_ranges = []

        for candidate in scored_candidates[:MAX_FINAL_CLIPS]:
            # Simplified boundary selection for resume
            chunk_start_line = candidate['chunk_metadata']['start_line']
            chunk_end_line = candidate['chunk_metadata']['end_line']
            
            all_words = []
            for seg in whisper_result['segments']:
                for word in seg['words']:
                    all_words.append(word)
            
            # Find approximate time range for this chunk
            words_in_chunk = all_words[chunk_start_line-1:chunk_end_line-1]
            if not words_in_chunk:
                continue
                
            proposed_start = words_in_chunk[0]['start']
            proposed_end = words_in_chunk[-1]['end']
            proposed_duration = proposed_end - proposed_start

            # Basic duration adjustment
            if proposed_duration > TARGET_CLIP_DURATION[1]:
                # Trim from both ends equally
                excess = proposed_duration - TARGET_CLIP_DURATION[1]
                proposed_start += excess / 2
                proposed_end -= excess / 2
            elif proposed_duration < TARGET_CLIP_DURATION[0]:
                # Skip if too short
                continue

            # Check for overlap
            overlaps = any(not (proposed_end < existing_start or proposed_start > existing_end) 
                         for (existing_start, existing_end) in selected_time_ranges)
            
            if not overlaps:
                selected_time_ranges.append((proposed_start, proposed_end))
                final_clips_metadata.append({
                    'start_time': proposed_start,
                    'end_time': proposed_end,
                    'metadata': candidate
                })

        # --- STAGE 5: Clipping & Packaging ---
        print("[5/5] Generating final clips and subtitles...")
        analysis_summary = []
        for i, clip_meta in enumerate(final_clips_metadata):
            start, end = clip_meta['start_time'], clip_meta['end_time']
            clip_filename = f"{base_filename}_clip_{i+1:02d}"
            output_video_path = os.path.join(OUTPUT_DIR, f"{clip_filename}.mp4")
            output_srt_path = os.path.join(OUTPUT_DIR, f"{clip_filename}.srt")

            cut_video_segment(video_file_path, output_video_path, start, end)
            # cut_video_segment_moviepy(video_file_path, output_video_path, start, end)
            create_subtitle_file(whisper_result, start, end, output_srt_path)

            analysis_summary.append({
                'clip_filename': f"{clip_filename}.mp4",
                'start_time': start,
                'end_time': end,
                'scores': clip_meta['metadata']['scores'],
                'overall_score': clip_meta['metadata']['overall_score']
            })
            print(f"  Created: {clip_filename}.mp4 (.srt)")

        summary_path = os.path.join(OUTPUT_DIR, f"{base_filename}_analysis.json")
        with open(summary_path, 'w') as f:
            json.dump(analysis_summary, f, indent=2)

        print(f"[+] Pipeline complete! Check the '{OUTPUT_DIR}' directory.")