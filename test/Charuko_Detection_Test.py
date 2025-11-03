import cv2
import cv2.aruco as aruco
import numpy as np
import glob
# --- 1. Define Board and Dictionary ---
import pandas as pd
# Use the dictionary from your JSON file: "Aruco5x5_1000"
ARUCO_DICT = aruco.DICT_5X5_1000

# Board specs from your JSON
SQUARES_X = 8
SQUARES_Y = 8
SQUARE_LENGTH = 0.1  # In meters
MARKER_LENGTH = 0.075 # In meters
dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
board = aruco.CharucoBoard(
    size=(SQUARES_X, SQUARES_Y),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=dictionary
)
arr = [['Image_Name','Aruco_Markers','ChArUco_Corners']]
aruco_params = aruco.DetectorParameters() 
aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
charuco_params = aruco.CharucoParameters()
detector = aruco.CharucoDetector(board, charuco_params, aruco_params)

IMAGE_DIR = "C:\\Users\\GirimaniKandan\\calibration\\calibration\\images"
files = glob.glob(f"{IMAGE_DIR}\\*.png")
print(f"Found {len(files)} images in {IMAGE_DIR}")
i = 0
for IMAGE_PATH in files:
    
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        exit()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # This single function does both marker detection AND corner interpolation
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_ids is not None and len(charuco_ids) > 0 and marker_corners is not None :
        nmarker_ids = len(marker_ids)
        ncharuko_ids = len(charuco_ids)
        
        aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
        aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids, (0, 255, 0))
    else :
        nmarker_ids = 0
        ncharuko_ids = 0

    arr.append([f'Image {i}',nmarker_ids , ncharuko_ids])
    i += 1
Df = pd.DataFrame(arr).to_csv('detection_summary.csv', index=False)
