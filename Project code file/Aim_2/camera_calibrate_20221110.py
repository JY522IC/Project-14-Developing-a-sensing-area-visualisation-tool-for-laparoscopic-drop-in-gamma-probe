#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/11/10 11:07
# @Author  : Yiyang
# @File    : camera_calibrate_20221110.py
# @Contact: jy522@ic.ac.uk
import pyrealsense2 as rs
import numpy as np
import cv2


config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline = rs.pipeline()

pipe_profile = pipeline.start(config)

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
# Intrinsics & Extrinsics
depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)
colorizer = rs.colorizer(2)
colorizer_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())


cv2.circle(colorizer_depth, (250, 250), 20, (0, 0, 255), 20)
cv2.imwrite(r'E:\Personal\First Term\Group Project\Code\Github\Project-14-Developing-a-sensing-area-visualisation-tool-for-laparoscopic-drop-in-gamma-probe\Project code file\Aim 2\depth.png', colorizer_depth)
# color_image = np.asanyarray(color_frame.get_data())
# cv2.imwrite(r'C:\Users\DELL\Desktop\color.png', color_image)

print ("\n Depth intrinsics: " + str(depth_intrin))
print ("\n Color intrinsics: " + str(color_intrin))
print ("\n Depth to color extrinsics: " + str(depth_to_color_extrin))

# Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
depth_sensor = pipe_profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
depth_scale = depth_sensor.get_option(rs.option.depth_units)
depth_image = np.asanyarray(depth_frame.get_data())
print ("\n\t depth_scale: " + str(depth_scale))
color_image = np.asanyarray(color_frame.get_data())

height = depth_image.shape[0]
width = depth_image.shape[1]
aligned_color = np.zeros((height, width, 3))
height_color = color_image.shape[0]
width_color = color_image.shape[1]

for v in range(width_color):
    for u in range(height_color):

        color_pixel = [v, u]
        color_point = rs.rs2_deproject_pixel_to_point(color_intrin, color_pixel, 1)
        depth_point = rs.rs2_transform_point_to_point(color_to_depth_extrin, color_point)
        depth_pixel = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)
        # depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), depth_scale,
        #     0.11, 1.0, depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, color_pixel)
        u_depth = int(round(depth_pixel[1]))
        v_depth = int(round(depth_pixel[0]))
        if u_depth < 0 or u_depth > height-1 or v_depth < 0 or v_depth > width-1:
            pass
        else:
            aligned_color[u_depth][v_depth][0] = color_image[u][v][0]
            aligned_color[u_depth][v_depth][1] = color_image[u][v][1]
            aligned_color[u_depth][v_depth][2] = color_image[u][v][2]
cv2.circle(aligned_color, (250, 250), 20, (0, 0, 255), 20)

cv2.imwrite(r'E:\Personal\First Term\Group Project\Code\Github\Project-14-Developing-a-sensing-area-visualisation-tool-for-laparoscopic-drop-in-gamma-probe\Project code file\Aim 2\aligned_color.png', aligned_color)
