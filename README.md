# DF_Linemod_ROS
**densefusion with ros using linemod format on custom dataset with chainer mask-rcnn for segmentation**
* tested on Ubuntu 18.04, ROS Melodic, RTX 2080-Ti, CUDA 10.1/10.2, Python3.6.9, PyTorch 1.5.1
* refer `environment.yml` for other anaconda packages
* git clone in your catkin_ws https://github.com/avasalya/DF_Linemod_ROS.git

## adapted from
* https://github.com/j96w/DenseFusion
* https://github.com/hygxy/ma_densefusion
* https://github.com/wkentaro/chainer-mask-rcnn

## create conda environment
* `conda env create -f environment.yml`
* install following lib manually
`open3d`,
`rospkg`,
`chainer_mask_rcn`,
`pyrealsense2`

## install realsense ROS package
* https://github.com/IntelRealSense/realsense-ros

## download and unzip `txonigiri` folder containing weights
* https://www.dropbox.com/sh/wkmqd0w1tvo4592/AADWt9j5SjiklJ5X0dpsSILAa?dl=0

<br />

# with ROS
### 1. launch camera
* `roslaunch realsense2_camera rs_rgbd.launch align_depth:=true`

### 2. launch rviz along with publisher/subscriber services
* change intrinsic parameters as per your camera from line #98 onwards
    [***cam_fx***, ***cam_fy***, ***cam_cx***, ***cam_cy***](https://github.com/avasalya/DF_Linemod_ROS/blob/c36b0f4527e654d176c0d4bce205f6bc8701ced4/scripts/densefusion_ros.py#L98)
* `roslaunch densefusion_ros densefusion.launch`
*  it publishes estimated pose as geometry_msgs/PoseArray and sensor_msgs/pointCloud2
*  also possible via
    * `rosrun densefusion_ros densefusion_ros.py` or
    * `./scripts/eval.sh` or
    * `python3 densefusion_ros.py`

<br />

# publish pose/pointCloud on ROS with `pyrealsense2` pkg
### 1. run `roscore`
### 2. `python3 densefusion_cam.py`
### 3. launch rviz `roslaunch densefusion_ros densefusion_cam.launch` (rgbd cloud not working YET!)


<br />

# Test with rgbd images [not updated]
`python3 densefusion_img.py`
