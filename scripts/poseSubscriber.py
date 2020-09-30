import rospy
import numpy as np
from colorama import Fore, Style
from geometry_msgs.msg import PoseArray


class poseSubscriber:

    def __init__(self):
        """ subscribe to pose in camera_color_optical_frame  """
        self.pose_sub = rospy.Subscriber('/onigiriPose', PoseArray, self.poseCallback, queue_size = 3)
        self.poseArray = PoseArray()


    def poseCallback(self, PoseArray):
        self.poseArray = PoseArray


    def Pose(self, getPose):
        msg2quat = np.zeros(4)
        msg2pos = np.zeros(3)

        for p in range(len(getPose)):
            msg2pos[0] = getPose[p].position.x
            msg2pos[1] = getPose[p].position.y
            msg2pos[2] = getPose[p].position.z
            msg2quat[0] = getPose[p].orientation.x
            msg2quat[1] = getPose[p].orientation.y
            msg2quat[2] = getPose[p].orientation.z
            msg2quat[3] = getPose[p].orientation.w
            print(f"{Fore.RED} pose obj{Style.RESET_ALL}", msg2pos)

        return msg2pos, msg2quat


def main():
    rospy.init_node('SubscribeOnigiriPose', anonymous=False)
    rospy.loginfo('streaming now...')

    ps = poseSubscriber()

    try:
        while True:
            header = ps.poseArray.header
            print(header)
            getPose = ps.poseArray.poses
            print(getPose)
            if header.seq > 2:
                msg2pos, msg2quat = ps.Pose(getPose)

    except rospy.ROSInterruptException:
        print(f'{Fore.RED}ROS Intruptted')


    rospy.spin()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main()