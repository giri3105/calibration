#!/usr/bin/env python

import numpy as np
import cv2 as cv
import cv2.aruco as aruco
import os

# --- 1. CONFIGURATION ---

# Path to the calibration file you saved
CALIBRATION_FILE = "/home/giri/calibration/debug/calibration_results.yml" # Or change to your debug dir
OUTPUT_VIDEO_PATH = "/home/giri/calibration/Results/pose_video_output.mp4"
# These MUST match the board you used for calibration
SQUARES_X = 9
SQUARES_Y = 7
SQUARE_LENGTH = 0.1  # In meters (100mm)
MARKER_LENGTH = 0.075 # In meters (75mm)
ARUCO_DICT_NAME = 'DICT_5X5_1000' # From your command


CIRCLE_RADIUS = 0.456 # (METERS) <-- UPDATE THIS VALUE

# We can calculate the circle's center based on your board size
# This assumes the grid is perfectly centered on the circle.
CIRCLE_CENTER_OFFSET_X = 0.45 #(SQUARES_X * SQUARE_LENGTH) / 2.0
CIRCLE_CENTER_OFFSET_Y = 0.40 #(SQUARES_Y * SQUARE_LENGTH) / 2.0

# Generate 50 points to make a smooth 3D circle
num_points = 50
theta = np.linspace(0, 2 * np.pi, num_points)
x = CIRCLE_CENTER_OFFSET_X + CIRCLE_RADIUS * np.cos(theta)
y = CIRCLE_CENTER_OFFSET_Y + CIRCLE_RADIUS * np.sin(theta)
z = np.zeros_like(x)

# This is our 3D model of the circle, (N, 3)
circle_object_points = np.stack([x, y, z], axis=-1)
# --- 2. LOAD CALIBRATION DATA ---

if not os.path.exists(CALIBRATION_FILE):
    print(f"Error: Calibration file not found at {CALIBRATION_FILE}")
    exit()

fs = cv.FileStorage(CALIBRATION_FILE, cv.FILE_STORAGE_READ)
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coefs = fs.getNode("distortion_coefficients").mat()
fs.release()

if camera_matrix is None or dist_coefs is None:
    print("Error: Failed to load camera matrix or distortion coefficients.")
    exit()

print("Calibration data loaded successfully.")

# --- 3. SETUP DETECTOR AND VIDEO ---

# Get the dictionary and create the board
try:
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, ARUCO_DICT_NAME))
except AttributeError:
    print(f"Error: Unknown ArUco dictionary: {ARUCO_DICT_NAME}")
    exit()

board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
detector = aruco.CharucoDetector(board)

# Start video capture
cap = cv.VideoCapture('/home/giri/calibration/Assets/videos/cam_front0-0-00.mp4') # Use 0 for webcam, or path to a video file
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
# Using 'mp4v' for .mp4 file
fourcc = cv.VideoWriter_fourcc(*'mp4v') 
out = cv.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print(f"Saving output video to: {OUTPUT_VIDEO_PATH}")

print("Starting video feed. Press 'q' to quit.")

# --- 4. POSE ESTIMATION LOOP ---

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # Convert to grayscale for detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect the board
    corners, ids, markerCorners, markerIds = detector.detectBoard(gray)
    
    # If we found enough corners, estimate pose
    if ids is not None and len(ids) > 4: # Need a few corners for a stable pose
        
        # This is the key function!
        # It finds the 3D rotation (rvec) and translation (tvec)
        # of the board relative to the camera.
        ret_pose, rvec, tvec = aruco.estimatePoseCharucoBoard(
            corners, 
            ids, 
            board, 
            camera_matrix, 
            dist_coefs, 
            None, # rvec (output)
            None  # tvec (output)
        )
        
        # If pose estimation was successful, draw the axes
        if ret_pose:
            # Draw a 3D axis on the board
            # The 0.1 means the axes will be 10cm long (same as square size)
            cv.drawFrameAxes(frame, camera_matrix, dist_coefs, rvec, tvec, SQUARE_LENGTH)

            projected_circle_points, _ = cv.projectPoints(
                circle_object_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coefs
            )

            # Draw the projected circle as a yellow polyline
            cv.polylines(
                frame,
                [np.int32(projected_circle_points)], # Needs to be int
                isClosed=True,
                color=(0, 255, 255), # Yellow
                thickness=2,
                lineType=cv.LINE_AA
            )
        
        # Draw the detected corners on the image
        aruco.drawDetectedCornersCharuco(frame, corners, ids)

    # Display the result
    cv.imshow('Pose Estimation', frame)

    # Quit on 'q'
    if cv.waitKey(1) == ord('q'):
        break

# --- 5. CLEANUP ---
cap.release()
cv.destroyAllWindows()
print("Script finished.")