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

    # Markers used
    marker_detected = 2 
    # marker type = 1, Blob type
    # marker type = 2, Aruco type

    cam_cal = np.load("calibration/calibration_realsense_D435i.npz.npz")

    if marker_detected == 1:
        # Load camera calibration
        camera_matrix = cam_cal['camera_matrix']
        dist_coef = cam_cal['dist_coef']

        # Instantiate marker detector
        mark = BlobMarker(camera_matrix, dist_coef)

    else:
        # Load camera calibration
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
            if marker_detected == 1:
                image = mark.detect_and_display_boundary(frame)
                text_0 = "Using Blob markers"
                # draw the ArUco marker ID on the image
                cv2.putText(image, text_0, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

            else:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image,central_points = mark.detect_and_display_boundary(frame)
                text_0 = "Using Aruco markers"
                # draw the ArUco marker ID on the image
                cv2.putText(image, text_0, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

            # Show image
            cv2.imshow('Detect Markers', image)

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