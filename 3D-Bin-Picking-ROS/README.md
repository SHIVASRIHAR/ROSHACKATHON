# Contents:  
* [1. Introduction](#1-introduction)
* [2. Environment Setup(**Mandatory**)](#2-Environment-Setup)
* [3. Grasping Simulations (***FINAL***)](#3-Grasping-Simulations)
    * [3.1 Building the Workspace](#31-Build-the-whole-workspace)
    * [3.2 Color Cube Grasping Demo](#32-Color-Cube-Grasping-Demo)
    * [3.3 Custom Object Grasping Demo](#33-custom-object-Grasping-Demo)

# 1. Introduction
   &ensp;&ensp;This repository contains the 3D Bin Picking SOlutions and models of xArm series and packages for ROS simulations.Recommeded Testing environment: Ubuntu 20.04 + ROS Noetic.  
   
# 2. Environment Setup

## 2.1 Install ROS and dependent package modules

ROS Noetic: <http://wiki.ros.org/noetic/Installation/Ubuntu>
Gazebo_ros_pkgs: <http://gazebosim.org/tutorials?tut=ros_installing>    
Ros_control: <http://wiki.ros.org/ros_control> 
Moveit_core: <https://moveit.ros.org/install/>  
   
## 2.2 Official tutorial documents for Reference
ROS Wiki: <http://wiki.ros.org/>  
Gazebo Tutorial: <http://gazebosim.org/tutorials>  
Gazebo ROS Control: <http://gazebosim.org/tutorials/?tut=ros_control>  
Moveit tutorial: <http://docs.ros.org/kinetic/api/moveit_tutorials/html/>
YoloV5: <https://github.com/ultralytics/yolov5>

# 3. Grasping Simulations
Simulations are based on Xarm5,Intel RealSense D435i depth camera.
## 3.1 Build the whole workspace：
```bash
$ catkin_make
``` 

## 3.2 Color Cube Grasping Demo

### 3.2.1 Download 'gazebo_grasp_plugin' for successful grasp simulation

 [Clone this Source File]<https://github.com/JenniferBuehler/gazebo-pkgs.git>

### 3.2.2 Gazebo grasping simulation
```bash
 # Initialize gazebo scene and move_group:
 $ roslaunch xarm_gazebo xarm_camera_scene.launch robot_dof:=5

 # In another terminal, run the color recognition and grasping script:
 $ rosrun xarm_gazebo color_recognition.py
```

## 3.3 Custom Object Grasping Demo

### 3.3.1 Gazebo grasping simulation
```bash
 # Initialize gazebo scene and move_group:
 $ roslaunch xarm_gazebo xarm_camera_scene.launch robot_dof:=5

 # In another terminal, run the Detect.py and grasping script:
 $ /bin/python3 /home/ashie/catws/src/xarm_ros/xarm_vision/yolov5/detect.py 
 #replace pwd 
 ```
![alt text](https://github.com/Ack-Robotics/3D-Bin-Picking/blob/ROS/media/unknown.png)
 





