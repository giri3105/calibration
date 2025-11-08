import cv2
import imageio
import numpy as np
import pyrealsense2 as rs
import os
import shutil
import keyboard
import time  # --- CHANGED 1: Import the time library ---

def make_clean_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    else:
        if len(os.listdir(path_folder)) == 0:
            print(f"Directory {path_folder} already exists and is empty. Using it.")
            return
        
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == "y":
            shutil.rmtree(path_folder)
            os.makedirs(path_folder)
        else:
            print("Exiting. Please clear the directory or choose a new one.")
            exit()


def record_rgbd():
    save_path = "/home/giri/calibration/Assets/images/Raw_with_lap"
    make_clean_folder(save_path) 

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    image_counter = 0
    is_recording = False

    # --- CHANGED 2: Set up your FPS throttle ---
    target_save_fps = 10.0  # The number of pictures to save per second
    save_interval = 1.0 / target_save_fps  # The time to wait between saves (0.1s)
    last_save_time = time.time()  # Get the current time to start

    print("Camera started. Press 's' to start/stop recording. Press 'q' to quit.")
    print(f"(You do not need to have the window focused. Will save at {target_save_fps} FPS)")

    try:
        while True:
            # Check for global key presses FIRST
            if keyboard.is_pressed('s'):
                if not is_recording:
                    is_recording = True
                    last_save_time = time.time() # Reset timer on start
                    print("\n--- Recording STARTED ---")
                    
            if keyboard.is_pressed('q'):
                print("\n--- Quitting ---")
                break # Exit the while loop

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("Warning: Could not acquire color frame. Skipping...")
                continue

            color_image = np.asanyarray(color_frame.get_data()) # This is BGR

            # Get the current time for this frame
            current_time = time.time()

            # ADD a visual "REC" indicator
            preview_display = color_image.copy()
            if is_recording:
                cv2.circle(preview_display, (30, 30), 15, (0, 0, 255), -1) 
                cv2.putText(preview_display, 'REC', (55, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Preview (S=Start, Q=Quit)", preview_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n--- Quitting (from window) ---")
                break 

            # --- CHANGED 3: Modify the SAVE logic ---
            if is_recording:
                # Check if enough time (0.1s) has passed since the last save
                if (current_time - last_save_time) >= save_interval:
                    
                    # If yes, update the last save time to NOW
                    last_save_time = current_time 
                    
                    # And run all your save logic
                    image_counter += 1
                    
                    color_image_rgb = color_image[..., ::-1]
                    rgb_filename = os.path.join(save_path, f"rgb_{image_counter}.png")
                    imageio.imwrite(rgb_filename, color_image_rgb)
                    
                    # This print is helpful for debugging the FPS
                    # print(f"Saved image {image_counter}") 

    finally:
        if is_recording:
            print(f"\n--- Recording STOPPED. Saved {image_counter} images. ---")
            
        print("Stopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    record_rgbd()