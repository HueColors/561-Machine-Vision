import cv2
import numpy as np
import glob

# Set print options to avoid scientific notation and display floats in normal format
np.set_printoptions(suppress=True, precision=6)

# Define the dimensions of the checkerboard (inside corners)
CHECKERBOARD = (10, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare the object points (0,0,0), (1,0,0), (2,0,0), ....,(9,6,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Load all checkerboard images from a folder (update the path)
images = glob.glob(r'C:\Users\hnguy\Desktop\checkerboard_images\*.jpg')

for fname in images:
    img = cv2.imread(fname)
    
    # Store original image size
    original_size = img.shape[1], img.shape[0]  # (width, height)

    # Resize the image to make it smaller and fit the screen (e.g., scale to 30%)
    scale_percent = 30  # Adjust this value to change the size of the image (30% of original size)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    img_resized = cv2.resize(img, dim)
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners on the resized image
    ret, corners_resized = cv2.findChessboardCorners(gray_resized, CHECKERBOARD, None)

    # If found, add object points and image points (after refining them)
    if ret:
        objpoints.append(objp)

        # Refine the corners
        corners_refined_resized = cv2.cornerSubPix(gray_resized, corners_resized, (11, 11), (-1, -1), criteria)

        # Scale the corners back to original image size
        scale_factor_x = original_size[0] / width
        scale_factor_y = original_size[1] / height
        corners_refined_original = corners_refined_resized * [scale_factor_x, scale_factor_y]

        # Ensure corners are in the correct format (float32)
        corners_refined_original = np.array(corners_refined_original, dtype=np.float32)
        imgpoints.append(corners_refined_original)

        # Draw and display the corners
        cv2.drawChessboardCorners(img_resized, CHECKERBOARD, corners_resized, ret)
        cv2.imshow('img', img_resized)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Camera calibration using points at original resolution
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, original_size, None, None)

# Print out the intrinsic parameters in normal format
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Save the camera matrix and distortion coefficients to use later
np.savez('camera_calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
