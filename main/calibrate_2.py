import cv2
import numpy as np
import os

# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.DICT_5X5_1000
SQUARES_VERTICALLY = 8      # Number of squares in Y direction
SQUARES_HORIZONTALLY = 8    # Number of squares in X direction
SQUARE_LENGTH = 0.1         # In meters
MARKER_LENGTH = 0.075       # In meters

PATH_TO_YOUR_IMAGES = '/home/giri/calibration/images'


def calibrate_and_save_parameters():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, aruco_params)

    image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".png")]
    image_files.sort()  

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None 

    print(f"Starting detection on {len(image_files)} images...")

    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read {image_file}")
            continue
            
        if image_size is None:
            h, w = image.shape[:2]
            image_size = (w, h) 

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        
        if charuco_ids is not None and len(charuco_ids) > 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

    print(f"Detection complete. Using {len(all_charuco_ids)} valid frames for calibration.")

    if len(all_charuco_ids) < 4:
        print("Error: Not enough valid frames to calibrate. Need at least 4.")
        return

    print("Calibrating camera...")
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, 
        all_charuco_ids, 
        board, 
        image_size,  
        None, 
        None
    )
    
    print("Calibration successful. Reprojection Error:", retval)

    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
    print("Saved 'camera_matrix.npy' and 'dist_coeffs.npy'")

    print("Displaying undistorted images. Press any key to cycle, or 'q' to quit.")
    for image_file in image_files:
        image = cv2.imread(image_file)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
        
        h, w = image.shape[:2]
        if h > 540: 
            h, w = h // 2, w // 2
            image = cv2.resize(image, (w,h))
            undistorted_image = cv2.resize(undistorted_image, (w,h))
            
        comparison_image = np.hstack((image, undistorted_image))
        cv2.imshow('Original (Left) vs. Undistorted (Right)', comparison_image)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

calibrate_and_save_parameters()