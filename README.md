# Group 4 - Project

Developing a sensing area visualisation tool for laparoscopic drop in gamma probe.

## Project Log

### Week 3 (17/10/22-23/10/22)
2022/10/17 — 2022/10/23--

#### 2022/10/17 2pm-5pm
 - First meeting with project supervisor and TAs.
 - Created GitHub repository for project files and logs.
 - Created Word document for project inception.
 - Received project explanation and suggested approach by the TAs.
 - Searched aruco marker source code online to start learning how to implement.
 - Kaizhong gave a presentation for this project.
   The pdf version: https://github.com/JY522IC/Project-14-Developing-a-sensing-area-visualisation-tool-for-laparoscopic-drop-in-gamma-probe/blob/main/Documents%20before%20start/Introduction%20to%20group%20project.pdf

### 2022/10/18 2pm-5pm
  - Brainstormed and decided on project proposal outline.
  - Divided the introduction chapter among team members.
  - Started writing the project introduction for the proposal.
  - Gathered reference papers that are relevant for the project.
  - Started writing the project objectives.
  - Found implementation of Aruco marker for reference.
  - Read up Intel RealSense SDK documentation.

### 2022/10/20 10am-1pm
  - Improve the 1st and 2nd chapter for inception report
  - Start the 3rd part for inception report
  - Researched materials for the methodology chapter

### 2022/10/21 10am-1pm
  - Improved the first three parts for inception report
  - Started with the Project Managment part of the article
  - Had a meeting with TA Kaizhong and got some suggestions for inception report
  - This weekends' plan for inception report:
        1) Rewrite overview
        2) Add Ros plan for chapter 2 and 3
        3) Finish the 4th part of inception report  
        4) Reference

### 2022/10/24 2pm—5pm
  - Meet with Daniel 
  - Installed SDK for Intel realsense camera D435i 
  - Tested the code to get the image from camera in python 
  - Aruco function test on camera 

### 2022/10/25 2pm—5pm 
  - Read opencv online resources and tried blob detection 
  - Implemented Aruco Marker detection algorithm 
  - Developed github code management standard  

### 2022/10/27 10am—1pm
  - Modified the inception report according to suggestions from professor and TAS 
  - Tested Aruco Marker algorithms  
  - Implemented webcam camera calibration and modified to Realsense camera calibration 

### 2022/10/28 10am—1pm 
  - Try the 3D reconstruction of RGB-D camera for D435i 
  - Check the filter algorism for sample image we found 
  - Check and upload the inception report 

### 2022/10/31 2pm-6pm
  - We report the works done of last week on meeting with Prof Dan
  - We printed smaller aruco marker and paste it on probe
  - We test our aruco algorithm to detect small marker on probe, problem encountered trying to solve on next meeting

### 2022/11/1 2pm—6pm
  - Still working on the calibrate code 
  - Spend time discuss what we have learned in Image Guided Intervention course and what methods can be used in our group project 

### 2022/11/3 10am—1pm
  - We re-calibrate the camera with small aruco marker 
  - We tested camera with small aruco marker 
  - Testing and integrate code 

### 2022/11/4 10am—1pm
  - We solved aruco detection problem with adding focus lens in front of camera 
  - We tested that 150mm len works best with our camera 
  - Refact and rearrange code files for better understanding 
  
### 2022/11/07 2pm—6pm
  - We report the works done of last week on meeting with Kaizhong and Baoru. 
  - Discuss marker development progress and RGB-Depth image correspondence. 
  - Planning to start task 2 and 3 of the project. 
  - Recap the project research, experiment, and result so far. 

### 2022/11/08 2pm—6pm
  - Start working on depth camera representation. 
  - Modified 3D camera code to show marker in color space. 
  - Tried to create marker in depth space, to be continued on the next session. 
  - Research on Realsense colour and depth frame class for Python. 

### 2022/11/10 10am—1pm
  - Resolve error shifted 3D space representation. 
  - Tried to create pixel representation on the 3D space. 
  - Visualise identified markers as green plus signs in 3D space. 
  - Encountered positional error for the marker depth. 

### 2022/11/11 10am—1pm 
  - Figured out how to systematically get the depth values from image x,y coordinates. 
  - Resolve error of identified marker depth on the 3D space, with some x-y axis error. 
  - Code refactoring and folder structure naming changes. 
