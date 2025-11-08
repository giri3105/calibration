import cv2
import cv2.aruco as aruco
import numpy as np
import os

# # --- ChArUco Board Features from your JSON ---
# SQUARES_X = 8        # corner_width
# SQUARES_Y = 8        # corner_height
# SQUARE_LENGTH = 0.1  # checker_length (in meters)
# MARKER_LENGTH = 0.075 # marker_length (in meters)
# ARUCO_DICT = aruco.DICT_5X5_1000 # marker_dictionary: "Aruco5x5_1000"

# # --- Image Generation Parameters ---
# IMAGE_RESOLUTION_WIDTH = 1920  # You can adjust this resolution
# IMAGE_RESOLUTION_HEIGHT = 1080 # You can adjust this resolution
# MARGIN_SIZE = 200 # Pixels around the board

#For Laptop screen
SQUARES_X = 10      # Corners wide (9 squares)
SQUARES_Y = 7     
SQUARE_LENGTH = 0.0288   # (Calculated as 2.88 cm)
MARKER_LENGTH = 0.0216   # (Calculated as 2.16 cm)
ARUCO_DICT = aruco.DICT_5X5_1000
# --- Image Generation Parameters ---

IMAGE_RESOLUTION_WIDTH = 1480 # (9 squares * 160px) + (2 * 20px margin)
IMAGE_RESOLUTION_HEIGHT = 1000 # (6 squares * 160px) + (2 * 20px margin)
MARGIN_SIZE = 20  # A 20-pixel border around the board

SAVE_PATH = "test_charuco_board.png" # Output file name

def generate_charuco_board_image():
    # 1. Get the ArUco dictionary
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)

    # 2. Create the ChArUco Board object
    board = aruco.CharucoBoard(
        size=(SQUARES_X, SQUARES_Y),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary
    )

    # 3. Calculate image size needed to fit the board with margins
    # The board.generateImage function takes the size in pixels,
    # and automatically scales the board to fit, respecting marker/square ratio.
    # We want to give it ample space and then add padding later.
    
    # Calculate desired board image size to keep a good resolution for detection
    # A good heuristic is to make the board image roughly the size of your actual camera images
    # or slightly larger to ensure detail.
    board_width_pixels = IMAGE_RESOLUTION_WIDTH - 2 * MARGIN_SIZE
    board_height_pixels = IMAGE_RESOLUTION_HEIGHT - 2 * MARGIN_SIZE

    if board_width_pixels <= 0 or board_height_pixels <= 0:
        print("Error: Margins are too large for the specified image resolution.")
        print("Please reduce MARGIN_SIZE or increase IMAGE_RESOLUTION_WIDTH/HEIGHT.")
        return

    # 4. Generate the board image
    # The `board.generateImage` method is deprecated, but still common in older OpenCV versions.
    # If it fails, you'll need the newer `board.generateImage(outSize, img=None, marginSize=0, borderBits=1)`.
    # For simplicity, we'll try the older widely compatible approach first.
    board_image = board.generateImage((board_width_pixels, board_height_pixels))
    
    if board_image is None:
        print("Error: Could not generate board image. Your OpenCV version might need `board.generateImage((w,h), marginSize=0, borderBits=1)`.")
        # Try the newer syntax if the old one fails
        try:
            board_image = board.generateImage((board_width_pixels, board_height_pixels), None, 0, 1)
        except Exception as e:
            print(f"Failed with newer syntax too: {e}")
            return


    # 5. Create a larger canvas and place the board image in the center
    # This creates a white background image that's slightly larger than the board image,
    # simulating a picture where the board is a reasonable distance from the camera.
    full_image = np.full((IMAGE_RESOLUTION_HEIGHT, IMAGE_RESOLUTION_WIDTH, 3), 255, dtype=np.uint8) # White background

    # Calculate top-left corner to paste the board image
    start_y = MARGIN_SIZE
    start_x = MARGIN_SIZE

    # Paste the board onto the full image
    full_image[start_y : start_y + board_image.shape[0], 
               start_x : start_x + board_image.shape[1]] = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)

    # 6. Save the generated image
    cv2.imwrite(SAVE_PATH, full_image)
    print(f"Generated ChArUco test board image saved to: {os.path.abspath(SAVE_PATH)}")

if __name__ == "__main__":
    generate_charuco_board_image()