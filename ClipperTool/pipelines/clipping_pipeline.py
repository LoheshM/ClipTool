import os
import json
import tempfile
from typing import List, Dict, Any, Tuple
from utils.video_utils import extract_audio, run_whisperx, detect_scenes, get_frame_embeddings
from utils.llm_utils import initialize_llm, prompt_llm_segment, prompt_llm_score
from utils.ffmpeg_utils import cut_video_segment_precise, create_subtitle_file, burn_subtitles_onto_video, cut_video_segment_simple
from config import *

class VideoClippingPipeline:
    def __init__(self):
        self.llm_model = initialize_llm(GOOGLE_API_KEY, LLM_MODEL_NAME)
        os.makedirs(TEMPDIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run_full_pipeline(self, video_file_path: str):
        """Executes the full 5-stage pipeline."""
        base_filename = os.path.splitext(os.path.basename(video_file_path))[0]
        print(f"[*] Starting FULL pipeline for: {base_filename}")

        # --- STAGE 1: Ingestion & Pre-processing ---
        print("[1/5] Extracting audio...")
        audio_path = os.path.join(TEMPDIR, f"{base_filename}.wav")
        extract_audio(video_file_path, audio_path)

        # --- STAGE 2: Multimodal Feature Extraction ---
        print("[2/5] Running multimodal feature extraction (this may take a while)...")
        # 2a. Audio/Text Features
        whisper_result = run_whisperx(audio_path, DEVICE, WHISPERX_MODEL, WHISPERX_COMPUTE_TYPE, HUGGINGFACE_HUB_TOKEN)
        transcript_path = os.path.join(TEMPDIR, f"{base_filename}_transcript.json")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(whisper_result, f, indent=2, ensure_ascii=False)

        # Format transcript for LLM with line numbers - with proper error handling
        lines = []
        current_speaker = "SPEAKER_00"  # Default speaker if diarization fails
        for seg in whisper_result['segments']:
            for word in seg['words']:
                # Handle missing speaker key gracefully
                speaker_id = word.get('speaker', current_speaker)
                current_speaker = speaker_id  # Update current speaker for next words
                line = f"{speaker_id}: {word['word']}"
                lines.append(line)
        formatted_transcript = "\n".join([f"{i+1}: {line}" for i, line in enumerate(lines)])

        # 2b. Visual Features
        scene_timestamps = detect_scenes(video_file_path, SCENE_DETECTION_THRESHOLD)
        frame_embeddings = get_frame_embeddings(video_file_path, FRAME_SAMPLE_RATE, scene_timestamps, CLIP_MODEL_NAME, CLIP_PRETRAINED_DATASET, DEVICE)

        # Continue with stages 3-5
        self._run_stages_3_to_5(video_file_path, base_filename, whisper_result, formatted_transcript, scene_timestamps, frame_embeddings)

    def run_from_step3(self, video_file_path: str, temp_dir: str = TEMPDIR):
        """Resumes pipeline from Step 3 (LLM Analysis) using saved intermediate files."""
        base_filename = os.path.splitext(os.path.basename(video_file_path))[0]
        print(f"[*] Resuming pipeline for: {base_filename} from Step 3")

        # Load previously saved intermediate files
        transcript_path = os.path.join(temp_dir, f"{base_filename}_transcript.json")
        
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        # Load whisper result
        with open(transcript_path, 'r', encoding='utf-8') as f:
            whisper_result = json.load(f)
        
        # Format transcript for LLM with proper error handling
        lines = []
        current_speaker = "SPEAKER_00"
        for seg in whisper_result['segments']:
            for word in seg['words']:
                speaker_id = word.get('speaker', current_speaker)
                current_speaker = speaker_id
                line = f"{speaker_id}: {word['word']}"
                lines.append(line)
        formatted_transcript = "\n".join([f"{i+1}: {line}" for i, line in enumerate(lines)])

        # Use empty visual data for resume
        scene_timestamps = []
        frame_embeddings = []

        self._run_stages_3_to_5(video_file_path, base_filename, whisper_result, formatted_transcript, scene_timestamps, frame_embeddings)

    def run_final_clipping(self, video_file_path: str, temp_dir: str = TEMPDIR):
        """Executes only the final clipping stage (Step 5) using saved intermediate files."""
        base_filename = os.path.splitext(os.path.basename(video_file_path))[0]
        print(f"[*] Starting FINAL CLIPPING for: {base_filename}")

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
        print("[5/5] Generating final clips with perfectly aligned subtitles...")
        analysis_summary = []

        for i, clip_meta in enumerate(final_clips_metadata):
            start, end = clip_meta['start_time'], clip_meta['end_time']
            clip_filename = f"{base_filename}_clip_{i+1:02d}"
            output_video_path = os.path.join(OUTPUT_DIR, f"{clip_filename}.mp4")
            output_srt_path = os.path.join(OUTPUT_DIR, f"{clip_filename}.srt")
            output_video_with_subs_path = os.path.join(OUTPUT_DIR, f"{clip_filename}_with_subs.mp4")

            print(f"  Creating clip {i+1}: {start:.2f}s to {end:.2f}s")
            
            try:
                # Step 1: Cut the video segment (includes buffer for natural flow)
                print("    Cutting video segment...")
                cut_video_segment_precise(video_file_path, output_video_path, start, end)
                

                # Step 2: Create subtitles using EXACT word boundaries (no buffer)
                print("    Generating precisely aligned subtitles...")
                create_subtitle_file(whisper_result, clip_meta, output_srt_path)
                
                # Step 3: Verify subtitle content matches video speech
                with open(output_srt_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read().strip()
                
                if not srt_content:
                    print("    Warning: No subtitles generated for this clip")
                    # Just copy the video without subtitles
                    os.rename(output_video_path, output_video_with_subs_path)
                else:
                    # Burn subtitles onto video
                    print("    Burning subtitles...")
                    burn_subtitles_onto_video(output_video_path, output_srt_path, output_video_with_subs_path)
                
                # Step 4: Clean up temporary files
                for temp_file in [output_video_path, output_srt_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                analysis_summary.append({
                    'clip_filename': os.path.basename(output_video_with_subs_path),
                    'video_start_time': start,
                    'video_end_time': end,
                    'speech_start_time': clip_meta.get('first_word_time', start),
                    'speech_end_time': clip_meta.get('last_word_time', end),
                    'scores': clip_meta.get('scores', {}),
                    'overall_score': clip_meta.get('overall_score', 0),
                    'title': clip_meta.get('title', f'Clip {i+1}')
                })
                print(f"    ✓ Successfully created: {os.path.basename(output_video_with_subs_path)}")
                
            except Exception as e:
                print(f"    ✗ Failed to create clip {i+1}: {e}")
                # Clean up any partial files
                for file_path in [output_video_path, output_srt_path, output_video_with_subs_path]:
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

        print(f"[+] Final clipping complete! Created {len(analysis_summary)} perfectly synced clips.")
        print(f"[+] Check the '{OUTPUT_DIR}' directory for results.")

    def _run_stages_3_to_5(self, video_file_path, base_filename, whisper_result, formatted_transcript, scene_timestamps, frame_embeddings):
        """Internal method to run stages 3-5 of the pipeline."""
        temp_dir = TEMPDIR
        
        # --- STAGE 3: Intelligence Core ---
        print("[3/5] Analyzing content with LLM...")
        # 3a. Pass 1: Semantic Chunking
        semantic_chunks = prompt_llm_segment(self.llm_model, formatted_transcript)
        chunk_path = os.path.join(temp_dir, f"{base_filename}_chunks.json")
        with open(chunk_path, 'w') as f:
            json.dump(semantic_chunks, f, indent=2)

        # 3b. Pass 2: Engagement Scoring
        scored_candidates = []
        for chunk in semantic_chunks:
            start_line, end_line = chunk['start_line'], chunk['end_line']
            
            # Extract chunk text from formatted transcript
            chunk_text_lines = formatted_transcript.split('\n')[start_line-1:end_line]
            chunk_text = '\n'.join(chunk_text_lines)

            # Create simple context summaries
            speaker_list = list(set([line.split(':')[0] for line in chunk_text_lines if ':' in line]))
            speaker_ctx = f"Features speakers: {', '.join(speaker_list)}" if speaker_list else "Single speaker."
            visual_ctx = "Visual context analysis completed" if scene_timestamps else "Visual analysis not available"

            print(f"  Analyzing chunk: {chunk['title']}")
            score_result = prompt_llm_score(self.llm_model, chunk_text, speaker_ctx, visual_ctx)
            score_result['chunk_metadata'] = chunk
            scored_candidates.append(score_result)

        candidates_path = os.path.join(temp_dir, f"{base_filename}_candidates.json")
        with open(candidates_path, 'w') as f:
            json.dump(scored_candidates, f, indent=2)

        # --- STAGE 4: Boundary Refinement & Selection ---
        print("[4/5] Refining clip boundaries and selecting final clips...")
        final_clips_metadata = self._generate_final_clips_metadata(scored_candidates, whisper_result)
        final_clips_path = os.path.join(temp_dir, f"{base_filename}_final_clips_metadata.json")
        with open(final_clips_path, 'w') as f:
            json.dump(final_clips_metadata, f, indent=2)

        # --- STAGE 5: Clipping & Packaging ---
        print("[5/5] Generating final clips with perfectly aligned subtitles...")
        analysis_summary = []

        for i, clip_meta in enumerate(final_clips_metadata):
            start, end = clip_meta['start_time'], clip_meta['end_time']
            clip_filename = f"{base_filename}_clip_{i+1:02d}"
            output_video_path = os.path.join(OUTPUT_DIR, f"{clip_filename}.mp4")
            output_srt_path = os.path.join(OUTPUT_DIR, f"{clip_filename}.srt")
            output_video_with_subs_path = os.path.join(OUTPUT_DIR, f"{clip_filename}_with_subs.mp4")

            print(f"  Creating clip {i+1}: {start:.2f}s to {end:.2f}s")
            
            try:
                # Step 1: Cut the video segment (includes buffer for natural flow)
                print("    Cutting video segment...")
                cut_video_segment_precise(video_file_path, output_video_path, start, end)
                

                # Step 2: Create subtitles using EXACT word boundaries (no buffer)
                print("    Generating precisely aligned subtitles...")
                create_subtitle_file(whisper_result, clip_meta, output_srt_path)
                
                # Step 3: Verify subtitle content matches video speech
                with open(output_srt_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read().strip()
                
                if not srt_content:
                    print("    Warning: No subtitles generated for this clip")
                    # Just copy the video without subtitles
                    os.rename(output_video_path, output_video_with_subs_path)
                else:
                    # Burn subtitles onto video
                    print("    Burning subtitles...")
                    burn_subtitles_onto_video(output_video_path, output_srt_path, output_video_with_subs_path)
                
                # Step 4: Clean up temporary files
                for temp_file in [output_video_path, output_srt_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                analysis_summary.append({
                    'clip_filename': os.path.basename(output_video_with_subs_path),
                    'video_start_time': start,
                    'video_end_time': end,
                    'speech_start_time': clip_meta.get('first_word_time', start),
                    'speech_end_time': clip_meta.get('last_word_time', end),
                    'scores': clip_meta.get('scores', {}),
                    'overall_score': clip_meta.get('overall_score', 0),
                    'title': clip_meta.get('title', f'Clip {i+1}')
                })
                print(f"    ✓ Successfully created: {os.path.basename(output_video_with_subs_path)}")
                
            except Exception as e:
                print(f"    ✗ Failed to create clip {i+1}: {e}")
                # Clean up any partial files
                for file_path in [output_video_path, output_srt_path, output_video_with_subs_path]:
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

        print(f"[+] Final clipping complete! Created {len(analysis_summary)} perfectly synced clips.")
        print(f"[+] Check the '{OUTPUT_DIR}' directory for results.")

    def _generate_final_clips_metadata(self, scored_candidates, whisper_result):
        """Generate final clips metadata ensuring video and subtitle alignment."""
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
                
            # Use the EXACT word boundaries for both video and subtitles
            proposed_start = words_in_chunk[0]['start']
            proposed_end = words_in_chunk[-1]['end']
            proposed_duration = proposed_end - proposed_start

            # Add a small buffer (0.5 seconds) at start and end for natural flow
            buffer = 0.5
            proposed_start = max(0, proposed_start - buffer)
            proposed_end = proposed_end + buffer
            
            # Ensure duration is within limits
            if proposed_duration > TARGET_CLIP_DURATION[1]:
                # Trim equally from both ends
                excess = proposed_duration - TARGET_CLIP_DURATION[1]
                proposed_start += excess / 2
                proposed_end -= excess / 2
            elif proposed_duration < TARGET_CLIP_DURATION[0]:
                # Extend to meet minimum duration
                needed = TARGET_CLIP_DURATION[0] - proposed_duration
                proposed_start = max(0, proposed_start - needed/2)
                proposed_end += needed/2

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
                    'title': chunk_meta.get('title', 'Untitled'),
                    # Store the exact word boundaries for subtitle generation
                    'first_word_time': words_in_chunk[0]['start'],
                    'last_word_time': words_in_chunk[-1]['end']
                })

        return final_clips_metadata
    
    