#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/11/3 11:06
# @Author  : Yiyang
# @File    : Blob detector RGB-D.py
# @Contact: jy522@ic.ac.uk

# Configure depth and color streams
import pyrealsense2 as rs
import numpy as np
import cv2
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)

# set aruco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 50
params.maxArea = 2000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.85

params.minRepeatability = 2
params.minCircularity = 0.75

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
detector = cv2.SimpleBlobDetector_create(params)

time.sleep(2)
cv2.namedWindow('blob',cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        keypoints = detector.detect(color_image)
        with_keypoints = cv2.drawKeypoints(color_image, keypoints, np.array([]),
        (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints",with_keypoints)
        if cv2.waitKey(5)&0xFF == ord('q'):
            break
            cv2.destroyAllWindows()
finally:
        # Stop streaming
        pipeline.stop()