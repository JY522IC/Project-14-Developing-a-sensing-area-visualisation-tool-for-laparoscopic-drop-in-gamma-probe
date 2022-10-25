#!/bin/bash
import camera
import cv2
import numpy as np

from markers.aruco import ArucoMarker

if __name__ == "__main__":

    # Instantiate camera
    # cam = camera.RealsenseCamera() # intel realsense
    cam = camera.WebcamCamera() # webcam

    # Load camera calibration
    # cam_cal = np.load('calibration/calibration_realsense.npz') # TODO realsense
    cam_cal = np.load('calibration/calibration_webcam.npz')
    camera_matrix = cam_cal['camera_matrix']
    dist_coef = cam_cal['dist_coef']

    # Instantiate marker detector
    mark = ArucoMarker(camera_matrix, dist_coef)

    # Loop until escaped
    try:
        while True:
            # Get frame from the camera
            frame = cam.get_frame()
            
            # Detect markers
            # image = mark.detect_and_display_box(frame)
            image = mark.detect_and_display_pose(frame)

            # Show image
            cv2.imshow('Detect Markers', image)


            key = cv2.waitKey(100)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        cam.stop()