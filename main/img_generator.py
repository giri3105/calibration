import cv2
import os
import argparse
import sys

def main():
    # --- 1. Setup command-line argument parser ---
    parser = argparse.ArgumentParser(description="Extract frames from a video file for calibration.")
    
    parser.add_argument("-v", "--video", 
                        required=True, 
                        help="Path to the input video file.")
                        
    parser.add_argument("-o", "--output_dir", 
                        required=True, 
                        help="Folder to save the output images (will be created if it doesn't exist).")
                        
    parser.add_argument("-s", "--skip", 
                        type=int, 
                        default=20, 
                        help="Number of frames to skip between saves. Saves 1 frame every 'skip' frames.")
                        
    parser.add_argument("-p", "--prefix", 
                        default="frame", 
                        help="Prefix for saved image files (e.g., 'frame' -> frame_0001.png).")
    
    # NEW: Add start and end time arguments
    parser.add_argument("--start_sec", 
                        type=float, 
                        default=0, 
                        help="Start time in seconds (e.g., 25.0 for 0:25).")
                        
    parser.add_argument("--end_sec", 
                        type=float, 
                        default=None, 
                        help="End time in seconds (e.g., 160.0 for 2:40).")

    args = parser.parse_args()

    # --- 2. Get and validate arguments ---
    video_path = args.video
    output_dir = args.output_dir
    frame_skip = args.skip
    prefix = args.prefix
    
    # NEW: Get time arguments
    start_msec = args.start_sec * 1000
    end_msec = args.end_sec * 1000 if args.end_sec is not None else None

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    if frame_skip <= 0:
        print("Error: --skip value must be 1 or greater.")
        sys.exit(1)

    # --- 3. Create output directory ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            sys.exit(1)
    else:
        print(f"Output directory {output_dir} already exists. Files may be overwritten.")

    # --- 4. Open video and process ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    # NEW: Set the starting position in the video
    if start_msec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)
        print(f"Seeking to {args.start_sec} seconds...")

    frame_count = 0
    saved_count = 0

    print(f"Starting frame extraction... saving 1 frame every {frame_skip} frames.")
    
    while True:
        # NEW: Check if we've passed the end time
        if end_msec is not None:
            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if current_msec > end_msec:
                print(f"Reached end time ({args.end_sec} seconds). Stopping.")
                break
        
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        # Check if this is a frame we should save
        # (We use the original frame_count to respect the --skip flag)
        if frame_count % frame_skip == 0:
            filename = f"{prefix}_{saved_count:04d}.png"
            save_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(save_path, frame)
            # print(f"Saved: {save_path}") # This can be noisy, optionally comment it out
            saved_count += 1
        
        frame_count += 1

    # --- 5. Clean up and finish ---
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Extracted {saved_count} frames from the specified time range to {output_dir}.")

if __name__ == '__main__':
    main()