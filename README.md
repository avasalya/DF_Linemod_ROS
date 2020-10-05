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

## change intrinsic parameters as per your camera, whichever scripts you use
* [***cam_fx***, ***cam_fy***, ***cam_cx***, ***cam_cy***](https://github.com/avasalya/DF_Linemod_ROS/blob/c36b0f4527e654d176c0d4bce205f6bc8701ced4/scripts/densefusion_ros.py#L98)
<br />

# with ROS
### 1. launch camera
* `roslaunch realsense2_camera rs_rgbd.launch align_depth:=true`

### 2. launch rviz along with publisher/subscriber services
* `roslaunch densefusion_ros densefusion.launch`
* `roslaunch densefusion_ros rviz.launch` [if rviz doesn't start automtically]
*  it publishes estimated pose as geometry_msgs/PoseArray
*  also possible via
    * `rosrun densefusion_ros densefusion_ros.py`
    * or `./scripts/eval.sh`
    * or `python3 scripts/densefusion_ros.py`

<br />

# publish pose/rgbd-pointCloud on ROS with `pyrealsense2` pkg
* `roscore`
* `python3 scripts/densefusion_cam.py`
*  launch rviz `roslaunch densefusion_ros densefusion_cam.launch`
*  Use `ESC` to stop
*  NOTE: `[pcd doesn't update, possible Rviz memory leaking issue]`


<br />

# Test with rgbd images [not updated]
`python3 scripts/densefusion_img.py`



<br />

# Known issues
* `Mask-rcnn is slow`, takes more than 0.5 seconds to process single frame, causes DF freezing with `densefusion_ros.py`. If this happens,
    * simply, restart the node again.
    * another solution, put a condition and receive new RGBD frame only when mask-rcnn has finished processing previous frame. --> this will increase overall inference time.
* if `no onigiri detected` msg appears, that would mean depth camera is unable to distinguish between object and background, in such case, change your object location.
* if using `densefusion_cam.py`, RGBD pointCloud won't update properly, due to probable RVIZ pointCloud memory leaking bug. Rely on 2D image output.
