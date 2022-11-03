#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/11/3 10:50
# @Author  : Yiyang
# @File    : Blob detector.py
# @Contact: jy522@ic.ac.uk

import cv2
import numpy as np
import time
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
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

capture = cv2.VideoCapture(0)
time.sleep(2)
cv2.namedWindow('blob',cv2.WINDOW_AUTOSIZE)

while(1):
  ret, frame = capture.read()
  im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  keypoints = detector.detect(im)
  with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]),
  (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.imshow("Keypoints",with_keypoints)
  if cv2.waitKey(5)&0xFF == ord('q'):
    break
capture.release()
cv2.destroyAllWindows()