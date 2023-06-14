#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState

import math
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim
import os

block_size = 24
columns = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device in use: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(block_size * columns, 256), # input
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

print(f"input size {block_size * columns}")


# replace variable with local path in linux
PATH = "classifier.pt"

### Load Existing Model
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(PATH))
optimizer = torch.optim.Adam()
optimizer.load_state_dict(torch.load(PATH))
model.eval() # must set dropout and batch normalization layers to evaluation mode

print("finished loading model")



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
    
    # In ROS, the jointstate.velocity field in a sensor_msgs/JointState message
    # represents the velocities of the joints at a particular time.
    # It is not an array that accumulates over time
    
    arr = np.array(JointState.velocity)
    print("--------------------------------------------")
    print(type(arr))
    print(arr.shape) # not iterable?
    
    arr = np.ndarray.flatten(arr)
    print(arr.shape)
    print("--------------------------------------------")
    # spin() simply keeps python from exiting until this node is stopped.
    # It does not interrupt the execution of the program
    y_pred = model(arr) # 1 or 0
    print(y_pred)



    rospy.spin()
    # no code goes after this
    

if __name__ == '__main__':
    listener()