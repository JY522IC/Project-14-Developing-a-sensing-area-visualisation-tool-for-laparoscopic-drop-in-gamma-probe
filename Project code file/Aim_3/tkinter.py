#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/11/18 10:29
# @Author  : Yiyang
# @File    : tkinter.py
# @Contact: jy522@ic.ac.uk
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x480 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        img = Image.fromarray(color_image)
        imgtk = ImageTk.PhotoImage(image=img)

        canvas = tk.Canvas(root, width=640, height=480)
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=imgtk)

finally:
    pipeline.stop()
