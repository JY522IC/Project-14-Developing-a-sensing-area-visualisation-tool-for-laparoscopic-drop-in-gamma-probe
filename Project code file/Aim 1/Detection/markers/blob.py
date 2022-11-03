import cv2
import numpy as np

BLOB_PARAMS = cv2.SimpleBlobDetector_Params()
# Change thresholds
BLOB_PARAMS.minThreshold = 10
BLOB_PARAMS.maxThreshold = 200

# Filter by Area.
BLOB_PARAMS.filterByArea = True
BLOB_PARAMS.minArea = 50
BLOB_PARAMS.maxArea = 2000

# Filter by Circularity
BLOB_PARAMS.filterByCircularity = True
BLOB_PARAMS.minCircularity = 0.1

# Filter by Convexity
BLOB_PARAMS.filterByConvexity = True
BLOB_PARAMS.minConvexity = 0.85

BLOB_PARAMS.minRepeatability = 2
BLOB_PARAMS.minCircularity = 0.75

# Filter by Inertia
BLOB_PARAMS.filterByInertia = True
BLOB_PARAMS.minInertiaRatio = 0.01


# Blob Marker Class
class BlobMarker:

    def __init__(self, camera_matrix, dist_coef):
        self.camera_matrix = camera_matrix
        self.dist_coef = dist_coef
        self.detector = cv2.SimpleBlobDetector_create(BLOB_PARAMS)

    def detect(self, image):
        return self.detector.detect(image)
    
    def detect_and_display_boundary(self, image):
        keypoints = self.detect(image)
        image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return image
    
    def detect_and_display_pose(self, image):
        # TODO: Implement this function

        return image

