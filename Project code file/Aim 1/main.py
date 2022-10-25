#!/bin/bash
import camera
import cv2

from markers.aruco import ArucoMarker

if __name__ == "__main__":

    # Instantiate camera
    # cam = camera.RealsenseCamera() # intel realsense
    cam = camera.WebcamCamera() # webcam

    # Instantiate marker detector
    mark = ArucoMarker()

    # Loop until escaped
    try:
        while True:
            # Get frame from the camera
            frame = cam.get_frame()
            
            # Detect markers
            image = mark.detect_and_display(frame)

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