import cv2

ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()


# Aruco Marker Class
class ArucoMarker:

    def __init__(self, camera_matrix, dist_coef):
        self.camera_matrix = camera_matrix
        self.dist_coef = dist_coef

    def detect(self, image):
        return cv2.aruco.detectMarkers(image, ARUCO_DICT, parameters=ARUCO_PARAMS)
    
    def detect_and_display_boundary(self, image):
        corners, ids, _ = self.detect(image)

        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                # compute and draw the center (x, y)-coordinates of the ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        return image
    
    def detect_and_display_pose(self, image):
        corners, ids, _ = self.detect(image)

        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            # estimate the pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.015, self.camera_matrix, self.dist_coef)

            # (rvec-tvec).any() # get rid of that nasty numpy value array error
            for (_, _, rvec, tvec) in zip(corners, ids, rvec, tvec):
                # draw axis for the aruco markers
                cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coef, rvec, tvec, 0.1)

        return image

