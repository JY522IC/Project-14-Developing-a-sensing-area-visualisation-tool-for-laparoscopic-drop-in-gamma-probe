import pyrealsense2 as rs
import numpy as np
import cv2

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    # set aruco dictionary and parameters
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
    aruco_params = cv2.aruco.DetectorParameters_create()

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

            # # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # # # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))
            # # # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)

            # images = np.hstack((color_image))
            # cv2.nameWindow('RealSense',cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', color_image)

            corners, ids, rejected = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=aruco_params)
            detected_markers = aruco_display(corners, ids, rejected, color_image)
            colorized_streams.append(detected_markers)



            key = cv2.waitKey(100)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()

