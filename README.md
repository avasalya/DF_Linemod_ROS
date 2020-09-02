# DF_Linemod_ROS
**densefusion with ros using format of linemod on custom dataset with chainer mask-rcnn for segmentation**

## Adapted from
* https://github.com/j96w/DenseFusion
* https://github.com/hygxy/ma_densefusion
* https://github.com/wkentaro/chainer-mask-rcnn

## Install conda environment
`conda env create -f environment.yml`

## 1. Test with ROS
* `roslaunch realsense2_camera rs_rgbd.launch align_depth:=true`

* `rosrun densefusion_ros default.launch`

* `rosrun densefusion_ros densefusion_ros.py --model=txonigiri`
    * #### also possible from directory  `./scripts/eval.sh` or `python3 densefusion_ros.py`

## 2. Test with realsense2_camera
`python3 densefusion_cam.py`

## 3. Test with rgbd images
`python3 densefusion_img.py`
