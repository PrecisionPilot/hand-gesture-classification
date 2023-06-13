#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.velocity)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/senseglove/0/rh/joint_states', JointState, callback) # subscribes to this senseglove joints topic
    arr = np.array(JointState.velocity)
    print("--------------------------------------------")
    print(JointState.velocity)
    print(type(arr))
    print(arr.shape)
    print("--------------------------------------------")
    arr = np.ndarray.flatten(arr)
    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()
    #y_pred = model(arr) # scalar
    #print(y_pred)

if __name__ == '__main__':
    listener()