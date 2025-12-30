
import os
from pipelines.clipping_pipeline import VideoClippingPipeline
import argparse


ffmpeg_bin_path = r"C:\Users\Lohesh\Documents\Packages\ffmpeg\bin"
os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ.get("PATH", "")

def main():
    parser = argparse.ArgumentParser(description='Automated Video Clipping Engine')
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('--stage', type=str, choices=['full', 'step3', 'final'], default='full',
                       help='Which stage to start from: full (1-5), step3 (LLM analysis), final (clipping only)')
    parser.add_argument('--temp-dir', type=str, default='./temp', help='Directory with intermediate files')
    
    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return

    pipeline = VideoClippingPipeline()

    try:
        if args.stage == 'full':
            print("=== Running FULL PIPELINE (Stages 1-5) ===")
            pipeline.run_full_pipeline(args.input_video)
            
        elif args.stage == 'step3':
            print("=== Resuming from STEP 3 (LLM Analysis) ===")
            pipeline.run_from_step3(args.input_video, args.temp_dir)
            
        elif args.stage == 'final':
            print("=== Running FINAL CLIPPING only (Stage 5) ===")
            pipeline.run_final_clipping(args.input_video, args.temp_dir)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the required intermediate files exist in the temp directory.")
        print("You may need to run an earlier stage first.")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()