#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/10/28 10:39
# @Author  : Yiyang
# @File    : point cloud reconstruction try.py
# @Contact: jy522@ic.ac.uk
# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer
This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.
Usage:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.
Keyboard:
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

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
    
    def detect_and_display_pose(self, image):
        image_copy = image.copy()
        corners, ids, _ = self.detect(image_copy)
        rvecs = []
        tvecs = []
        central_points = []
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            # estimate the pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.015, self.camera_matrix, self.dist_coef)
            rvecs.append(rvec)
            tvecs.append(tvec)

            # (rvec-tvec).any() # get rid of that nasty numpy value array error
            for (markerCorners, rvec, tvec) in zip(corners, rvec, tvec):
                # draw axis for the aruco markers
                cv2.drawFrameAxes(image_copy, self.camera_matrix, self.dist_coef, rvec, tvec, 0.1)

                corners = markerCorners.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                central_points.append([cX,cY])

        return image_copy, rvecs, tvecs, central_points


cam_cal = np.load("calibration_realsense2.npz")
camera_matrix = cam_cal['camera_matrix']
dist_coef = cam_cal['dist_coef']

# Instantiate marker detector
mark = ArucoMarker(camera_matrix, dist_coef)

axesPoints = np.float32([[0,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)

# Blob detecter
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

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
# print("Depth Scale is: " , depth_profile)
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


out = np.empty((h, w, 3), dtype=np.uint8)

while True:

    depth_intrinsics = None
    depth_image = None
    detected_image = None

    rvecs = None
    tvecs = None
    centralPoints = None

    # Grab camera data
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # Define the reason for color depth corresponse
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        detected_image, rvecs, tvecs, centralPoints = mark.detect_and_display_pose(color_image)
        # print(rvec,tvec)

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # Render
    now = time.time()

    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))


    ## ---------------------------------------------------------------------------------------------------------------##
    ## GUI design based on openCV

    colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'cyan': (255, 255, 0),
              'magenta': (255, 0, 255), 'yellow': (0, 255, 255), 'black': (0, 0, 0), 'white': (255, 255, 255),
              'gray': (125, 125, 125), 'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220),
              'rand': np.random.randint(0, high=256, size=(3,)).tolist()}

    # Draw the 3D line for the marker in space
    if centralPoints != []:
        w, h = depth_image.shape[1], depth_image.shape[0]
        for i in range(len(centralPoints)):
            try:
                centralPoint = centralPoints[i]
                rvec = rvecs[i]
                tvec = tvecs[i]
            except:
                continue
            x, y = centralPoint[0], centralPoint[1]
            x = int(x / (2**state.decimate))
            y = int(y / (2**state.decimate))
            
            try:
                image_points = cv2.projectPoints(axesPoints,rvec,tvec,camera_matrix,dist_coef)
                rotation_matrix = cv2.Rodrigues(rvec)[0]
                marker_depth = depth_image[int(y), int(x)] * depth_scale
                p = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], marker_depth)
                print(f"Marker depth: {marker_depth:.3f} m")
                if p[2] <= 0:
                    continue
                line3d(out, view(p), view(p) + np.dot((0, 0, 0.1), rotation_matrix), (0xff, 0, 0), 1)
                line3d(out, view(p), view(p) + np.dot((0, 0.1, 0), rotation_matrix), (0, 0xff, 0), 1)
                line3d(out, view(p), view(p) + np.dot((0.1, 0, 0), rotation_matrix), (0, 0, 0xff), 1)

                # Display probe distance on reconstruction image
                text_0 = "Probe Distance to Camera = " + f"{marker_depth:.3f}" + 'm'
                cv2.putText(detected_image, text_0, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['red'])

            except Exception as e:
                print(e)

        # Calculate the distance between the probe and the test point
        if len(centralPoints) == 2:
            centralPoint_1 = centralPoints[0]
            x_1, y_1 = centralPoint_1[0], centralPoint_1[1]
            x_1 = int(x_1 / (2**state.decimate))
            y_1 = int(y_1 / (2**state.decimate))

            centralPoint_2 = centralPoints[1]
            x_2, y_2 = centralPoint_2[0], centralPoint_2[1]
            x_2 = int(x_2 / (2**state.decimate))
            y_2 = int(y_2 / (2**state.decimate))

            try:
                marker_depth_1 = depth_image[int(y_1), int(x_1)] * depth_scale
                p_1 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_1, y_1], marker_depth_1)

                marker_depth_2 = depth_image[int(y_2), int(x_2)] * depth_scale
                p_2 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_2, y_2], marker_depth_2)

                disance_between_points = ((p_1[0] - p_2[0])** 2 + (p_1[1] - p_2[1])** 2 + (p_1[2] - p_2[2])** 2) ** (0.5)

                print(f'distance between 2 points {disance_between_points:.3f}m')

                # Display the relative disdance and relative postion on reconstruction image
                text_3 = f"The relative distance between two point {disance_between_points:.3f}m"
                cv2.putText(detected_image, text_3, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['blue'])

            except Exception as e:
                # print(e)
                pass

    out2 = np.hstack([out, detected_image])
    cv2.imshow(state.WIN_NAME, out2)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Stop streaming
pipeline.stop()