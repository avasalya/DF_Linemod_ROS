# DF_Linemod_ROS
densefusion with ros using format of linemod on custom dataset

# adapted from
https://github.com/hygxy/ma_densefusion 

# to test with ROS
$ roslaunch realsense2_camera rs_rgbd.launch align_depth:=true

$ rosrun densefusion_ros densefusion_ros.py --model=txonigiri

or just run from scripts/

$ ./eval.sh

or 

$ python3 densefusion_ros.py

# to test with realsense2_camera
$ python3 df-cmr-live.py

# to test with rgbd images
$ python3 df-cmr-eval.py
