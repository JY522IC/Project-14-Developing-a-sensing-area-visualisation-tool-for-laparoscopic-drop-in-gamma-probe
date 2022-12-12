"""
Aruco camera calibration
Adapted from: https://gist.githubusercontent.com/edward1986/b54c0a1e645ec1e82bb060ad0fcb8017/
              raw/861b29eeaf90e2861e05bc49a9fef0b7c3f92d9a/Aruco%20camera%20calibration.py
"""

import cv2
import numpy as np
import os
import sys

# Add camera module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import camera

# Create dictionary and board object
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(4, 5, .025, .005, dictionary)

# Create board image to be used in the calibration process
image_board = board.draw((200 * 4, 200 * 5))

# Write calibration board image
cv2.imwrite('charuco_small.png', image_board)

# Instantiate camera
cam = camera.RealsenseCamera() # intel realsense
# cam = camera.WebcamCamera() # webcam

all_corners = []
all_ids = []
counter = 0
for i in range(300):

    # Get frame from the camera
    frame = cam.get_frame()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and counter % 3 == 0:
            all_corners.append(res2[1])
            all_ids.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    counter += 1


# Calibration can fail for many reasons
try:
    cal = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
    print(cal)
except Exception as e:
    print(e)
    cam.stop()
    print("Calibration could not be done ...")

# Get the calibration result
_, camera_matrix, dist_coef, _, _ = cal

# Save the calibration result
np.savez('calibration_realsense2.npz', camera_matrix=camera_matrix, dist_coef=dist_coef)
# np.savez('calibration_webcam.npz', camera_matrix=camera_matrix, dist_coef=dist_coef)
# Release everything:
cv2.destroyAllWindows()