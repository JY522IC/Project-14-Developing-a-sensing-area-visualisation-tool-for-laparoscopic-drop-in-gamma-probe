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

### 2022/11/14 2pm—6pm 
  - Project meeting with Prof. Dan
  - Solved the x,y shift problem found last week
  - Changed the code to detect Aruco markers and display a plus sign on the centre of the Aruco markers in the 3D point cloud. 
  - Started working on camera calibration with lens 

### 2022/11/15 2pm—6pm 
  - Started working on the GUI 
  - Started working on pose estimation programs 
  - Researched RGB to D calibration with lens but didn't get solutions 
  - Discussed with TA Kaizhong and decided the plan: work on GUI, pose estimation and only calibrate the RGB camera  (might try to get a transformation matrix between results with lens and without lens) 

### 2022/11/17 10am—1pm
  - Looked up GUI libraries and discussed which to be used 
  - Decided how to visualize the pose 

### 2022/11/18 10am—1pm 
  - Worked on Aruco pose calculation 
  - Added GUI functions 
  - Finalised marker design 
  
### 2022/11/21 2pm-6pm
  - Project meeting with Prof.Dan
  - Looked up PPT template
  - Tried the 150mm lens
  
### 2022/11/22 2pm-6pm
  - Finished the PPT template 
  - Find the Aruco pose axis in color image 
  - Projected the direction axis to 3D space 
  
### 2022/11/24 10am-1pm
  - Started did the experiment
  - Checked the value get for depth
  
### 2022/11/25 10am-1pm
  - Continued with the experiment
  - Analysis the error exist so far

### 2022/11/28 2pm-6pm 
  - Meeting with Dr.Daniel 
  - Implement the problem we meet for find the depth and position of probe and coordinates 
  - Discussion with kaizhong about the calibration problem with lens 

### 2022/12/01 10am-1pm 
  - Finish making the PPT template
  - Try some way to fix the problem based on the original code

### 2022/12/02 10am-12am 
  - Look up the information about self-calibration using Pyrealsense camera
  
### 2022/12/03 1pm-6pm 
  - Using Intel RealSense Viewer for On-chip Calibration, Tare clibration, focus-length calibration
  
  
