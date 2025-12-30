import os
import json
from typing import List, Dict, Any
from utils.ffmpeg_utils import cut_video_segment, cut_video_segment_precise, create_subtitle_file
from config import *

class FinalClippingPipeline:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run_final_clipping(self, video_file_path: str, temp_dir: str = TEMPDIR):
        """Executes only the final clipping stage (Step 5) using saved intermediate files."""
        base_filename = os.path.splitext(os.path.basename(video_file_path))[0]
        print(f"[*] Starting final clipping for: {base_filename}")

        # Load all necessary intermediate files
        transcript_path = os.path.join(temp_dir, f"{base_filename}_transcript.json")
        candidates_path = os.path.join(temp_dir, f"{base_filename}_candidates.json")
        final_clips_path = os.path.join(temp_dir, f"{base_filename}_final_clips_metadata.json")
        
        # Check if files exist
        required_files = [transcript_path, candidates_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load transcript data
        with open(transcript_path, 'r', encoding='utf-8') as f:
            whisper_result = json.load(f)

        # Load scored candidates
        with open(candidates_path, 'r') as f:
            scored_candidates = json.load(f)

        # Try to load final clips metadata if it exists, otherwise create it
        if os.path.exists(final_clips_path):
            print("  Loading existing final clips metadata...")
            with open(final_clips_path, 'r') as f:
                final_clips_metadata = json.load(f)
        else:
            print("  Generating final clips metadata from candidates...")
            final_clips_metadata = self._generate_final_clips_metadata(scored_candidates, whisper_result)
            # Save for future use
            with open(final_clips_path, 'w') as f:
                json.dump(final_clips_metadata, f, indent=2)

    # --- STAGE 5: Clipping & Packaging ---
    print("[5/5] Generating final clips with perfect subtitle synchronization...")
    analysis_summary = []

    for i, clip_meta in enumerate(final_clips_metadata):
        start, end = clip_meta['start_time'], clip_meta['end_time']
        clip_filename = f"{base_filename}_clip_{i+1:02d}"
        output_video_path = os.path.join(OUTPUT_DIR, f"{clip_filename}.mp4")
        output_srt_path = os.path.join(OUTPUT_DIR, f"{clip_filename}.srt")
        output_video_with_subs_path = os.path.join(OUTPUT_DIR, f"{clip_filename}_with_subs.mp4")
        temp_audio_path = os.path.join(TEMPDIR, f"{clip_filename}_temp.wav")

        print(f"  Creating clip {i+1}: {start:.2f}s - {end:.2f}s")
        
        try:
            # Step 1: Cut the video segment
            print("    Cutting video...")
            cut_video_segment_precise(video_file_path, output_video_path, start, end)
            
            # Get video duration
            try:
                probe = ffmpeg.probe(output_video_path)
                video_duration = float(probe['format']['duration'])
            except:
                video_duration = end - start
            
            # Step 2: Extract audio from the clip
            print("    Extracting audio for transcription...")
            extract_audio(output_video_path, temp_audio_path)
            
            # Step 3: Transcribe the clip audio using WhisperX
            print("    Transcribing clip audio...")
            whisper_result = run_whisperx_on_clip(
                temp_audio_path, 
                DEVICE, 
                WHISPERX_CLIP_MODEL, 
                WHISPERX_CLIP_COMPUTE_TYPE, 
                HUGGINGFACE_HUB_TOKEN
            )
            
            # Step 4: Create subtitles from the clip transcription
            print("    Creating perfectly synced subtitles...")
            create_subtitle_file_grouped(whisper_result, output_srt_path)
            
            # Verify SRT file was created and has content
            srt_exists = os.path.exists(output_srt_path)
            srt_has_content = False
            if srt_exists:
                with open(output_srt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    srt_has_content = bool(content)
            
            if not srt_has_content:
                print("    No subtitles generated, creating placeholder...")
                create_subtitle_file_simple_fallback(video_duration, output_srt_path)
            
            # Step 5: Burn subtitles onto video using safe method
            print("    Burning subtitles (safe method)...")
            burn_subtitles_onto_video_safe(output_video_path, output_srt_path, output_video_with_subs_path)
            
            # Step 6: Clean up temporary files
            for temp_file in [temp_audio_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            analysis_summary.append({
                'clip_filename': os.path.basename(output_video_with_subs_path),
                'subtitle_filename': os.path.basename(output_srt_path),
                'video_start_time': start,
                'video_end_time': end,
                'video_duration': video_duration,
                'scores': clip_meta.get('scores', {}),
                'overall_score': clip_meta.get('overall_score', 0),
                'title': clip_meta.get('title', f'Clip {i+1}')
            })
            
            print(f"    ✓ Success: {os.path.basename(output_video_with_subs_path)}")
            print(f"    ✓ Subtitles: {os.path.basename(output_srt_path)}")
            
        except Exception as e:
            print(f"    ✗ Failed to create clip {i+1}: {e}")
            
            # Try to at least create the video without subtitles
            try:
                if os.path.exists(output_video_path) and not os.path.exists(output_video_with_subs_path):
                    os.rename(output_video_path, output_video_with_subs_path)
                    print(f"    ⚠️  Created video without subtitles: {os.path.basename(output_video_with_subs_path)}")
            except:
                pass
                
            # Clean up any partial files
            for file_path in [output_video_path, output_srt_path, output_video_with_subs_path, temp_audio_path]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            continue

    # Save the analysis summary
    summary_path = os.path.join(OUTPUT_DIR, f"{base_filename}_analysis.json")
    with open(summary_path, 'w') as f:
        json.dump(analysis_summary, f, indent=2)

    print(f"[+] Final clipping complete! Created {len(analysis_summary)} clips.")
    print(f"[+] Check the '{OUTPUT_DIR}' directory for results.")

    def _generate_final_clips_metadata(self, scored_candidates, whisper_result):
        """Generate final clips metadata from scored candidates."""
        # Sort candidates by overall score
        scored_candidates.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        
        final_clips_metadata = []
        selected_time_ranges = []
        
        # Extract all words for time calculation
        all_words = []
        for seg in whisper_result['segments']:
            for word in seg['words']:
                all_words.append(word)

        for candidate in scored_candidates[:MAX_FINAL_CLIPS]:
            chunk_meta = candidate.get('chunk_metadata', {})
            start_line = chunk_meta.get('start_line', 1)
            end_line = chunk_meta.get('end_line', len(all_words))
            
            # Get words for this chunk
            words_in_chunk = all_words[start_line-1:end_line]
            if not words_in_chunk:
                continue
                
            # Calculate time boundaries
            proposed_start = words_in_chunk[0]['start']
            proposed_end = words_in_chunk[-1]['end']
            proposed_duration = proposed_end - proposed_start

            # Adjust to target duration
            if proposed_duration > TARGET_CLIP_DURATION[1]:
                # Center the clip around the middle
                center = (proposed_start + proposed_end) / 2
                half_duration = TARGET_CLIP_DURATION[1] / 2
                proposed_start = max(0, center - half_duration)
                proposed_end = center + half_duration
            elif proposed_duration < TARGET_CLIP_DURATION[0]:
                # Skip if too short
                continue

            # Check for overlap
            overlaps = any(
                not (proposed_end < existing_start or proposed_start > existing_end)
                for (existing_start, existing_end) in selected_time_ranges
            )
            
            if not overlaps:
                selected_time_ranges.append((proposed_start, proposed_end))
                final_clips_metadata.append({
                    'start_time': round(proposed_start, 3),
                    'end_time': round(proposed_end, 3),
                    'scores': candidate.get('scores', {}),
                    'overall_score': candidate.get('overall_score', 0),
                    'title': chunk_meta.get('title', 'Untitled')
                })

        return final_clips_metadata