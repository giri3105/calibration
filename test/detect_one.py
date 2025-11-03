import cv2
import cv2.aruco as aruco
import numpy as np

# --- 1. Define Board and Dictionary ---

# Use the dictionary from your JSON file: "Aruco5x5_1000"
ARUCO_DICT = aruco.DICT_5X5_1000

# Board specs from your JSON
SQUARES_X = 8
SQUARES_Y = 8
SQUARE_LENGTH = 0.1  # In meters
MARKER_LENGTH = 0.075 # In meters

# !!! Change this to the path of your image !!!
IMAGE_PATH = "/home/giri/calibration/Sharpened (Unsharp Mask).png" 

# --- 2. Load the Image ---

# Load the image in color so we can draw on it
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Error: Could not load image from {IMAGE_PATH}")
    exit()

# Convert to grayscale for detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- 3. Setup the Detector and Board ---

# Get the predefined dictionary
dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)

# Create the ChArUco board object
board = aruco.CharucoBoard(
    size=(SQUARES_X, SQUARES_Y),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=dictionary
)

# --- THIS IS THE UPDATED PART ---

# Create detector parameters for the ArUco markers
aruco_params = aruco.DetectorParameters() 
# aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
# Create parameters for the ChArUco board detector
charuco_params = aruco.CharucoParameters()
# You can add logic like charuco_params.tryRefineMarkers = True

# Create the CharucoDetector object
detector = aruco.CharucoDetector(board, charuco_params, aruco_params)

# --- 4. Detect ChArUco Board ---

print("Detecting ChArUco board (markers and corners)...")

# This single function does both marker detection AND corner interpolation
charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

# --- 5. Display the Results ---

# Check if we found ChArUco corners
if charuco_ids is not None and len(charuco_ids) > 0:
    print(f"Success! Found {len(charuco_ids)} ChArUco corners.")
    print(f"Also found {len(marker_ids)} ArUco markers.")
    
    # Draw the ArUco markers (optional)
    aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
    
    # Draw the ChArUco corners
    aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids, (0, 255, 0))

# Check if we only found markers but no corners
elif marker_ids is not None and len(marker_ids) > 0:
    print(f"Found {len(marker_ids)} ArUco markers, but could not interpolate any ChArUco corners.")
    # Draw only the ArUco markers
    aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

else:
    print("Detection failed. No ArUco markers found.")

# --- 6. Display the Results ---

cv2.imshow("Detected ChArUco Corners", image)
print("Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()