import cv2
import numpy as np
import pyrealsense2 as rs


# Intel Realsense Pipeline
class RealsenseCamera:
    
    # Initialise class
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Run initialisation
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
    
    # Get frame
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())

        return color_image
    
    # Stop interface
    def stop(self):
        self.pipeline.stop()


# Webcam Interface
class WebcamCamera:
        
    # Initialise class
    def __init__(self, camera_id=0):
        self.camera = cv2.VideoCapture(camera_id)
    
    # Get frame
    def get_frame(self):
        _, frame = self.camera.read()
        return frame
    
    # Stop interface
    def stop(self):
        self.camera.release()
