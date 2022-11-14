#!/bin/bash
import camera
import cv2
import numpy as np

from markers.aruco import ArucoMarker
from markers.blob import BlobMarker

if __name__ == "__main__":

    # Instantiate camera
    cam = camera.RealsenseCamera() # intel realsense
    # cam = camera.WebcamCamera() # webcam

    # Load camera calibration
    cam_cal = np.load("D:\works\Powerpoint & PDF\Postgraduate_study\Group_project\group_project\Project code file\Aim_1\Detection\calibration\calibration_realsense.npz")
    # cam_cal = np.load('D:\works\Powerpoint & PDF\Postgraduate_study\Group_project\group_project\Project code file\Aim 1\Detection\calibration\calibration_webcam.npz')
    camera_matrix = cam_cal['camera_matrix']
    dist_coef = cam_cal['dist_coef']

    # Instantiate marker detector
    mark = ArucoMarker(camera_matrix, dist_coef)
    # mark = BlobMarker(camera_matrix, dist_coef)

    # Loop until escaped
    try:
        while True:
            # Get frame from the camera
            frame = cam.get_frame()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            # # Apply gaussian and log
            # gauss_kernel = cv2.getGaussianKernel(5, 0)
            # frame = cv2.filter2D(frame, -1, gauss_kernel)
            # log_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            # frame = cv2.filter2D(frame, -1, log_kernel)
            
            # Detect markers
            _,image = mark.detect_and_display_boundary(frame)
            if (len(image)!=0):
                cv2.circle(frame, (round(image[0][0]), round(image[0][1])), 20, (0, 0, 255), 20)
            # image = mark.detect_and_display_pose(frame)

            # Show image
            cv2.imshow('Detect Markers', frame)


            key = cv2.waitKey(100)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
    finally:
        # Stop streaming
        cam.stop()