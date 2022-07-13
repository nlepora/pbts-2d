"""Simple test script for AsyncRobot class using MG400Controller.
Note: Works in VSCode. Does not work in Spyder.
"""

import numpy as np, time

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller

np.set_printoptions(precision=2, suppress=True)


def main():
    base_frame = (0, 0, 0, 0, 0, 0)
    work_frame = (300, 38, 0, 0, 0, 0)  # base frame: x->front, y->left, z->up, rz->anticlockwise
    
    with SyncRobot(Controller()) as robot:
        # Set TCP, linear speed,  angular speed and coordinate frame
        robot.tcp = (0, -38, 0, 0, 0, 0) # not implemented
        robot.linear_speed = 100
        robot.angular_speed = 100 # not implemented
        robot.coord_frame = work_frame
        
        # Display robot info
        print("Robot info: {}".format(robot.info))

        # Display initial joint angles
        print("Initial joint angles: {}".format(np.asarray(robot.joint_angles)))

        # Display initial pose in work frame
        print("Initial pose in work frame: {}".format(robot.pose))
        
        # Move to origin of work frame
        print("Moving to origin of work frame ...")
        robot.move_linear((0, 0, 0, 0, 0, 0))
        
        print("Robot joint angles",robot.joint_angles)
        print("Robot pose: {}".format(robot.pose))

        # Increase and decrease all joint angles - does not work
        print("Increasing and decreasing all joint angles ...")
        robot.move_joints(robot.joint_angles + (10,)*4)   
        print("Target joint angles after increase: {}".format(robot.target_joint_angles))
        print("Joint angles after increase: {}".format(robot.joint_angles))
        robot.move_joints(robot.joint_angles - (10,)*4)  
        print("Target joint angles after decrease: {}".format(robot.target_joint_angles))
        print("Joint angles after decrease: {}".format(robot.joint_angles))
        
        # Move backward and forward
        print("Moving backward and forward ...")        
        robot.move_linear((-50, 0, 0, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))
        
        # Move right and left
        print("Moving right and left ...")  
        robot.move_linear((0, -50, 0, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))
        
        # Move down and up
        print("Moving down and up ...")  
        robot.move_linear((0, 0, -50, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Turn clockwise and anticlockwise around work frame z-axis
        print("Turning clockwise and anticlockwise around work frame z-axis ...")        
        robot.move_linear((0, 0, 0, 0, 0, -50))
        robot.move_linear((0, 0, 0, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 50))
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Move to offset pose then tap down and up in sensor frame
        print("Moving to 50 mm/ 30 deg offset in pose ...")         
        robot.move_linear((50, 50, 50, 0, 0, 30))
        print("Target pose after offset move: {}".format(robot.target_pose))
        print("Pose after offset move: {}".format(robot.pose))
        print("Tapping down and up ...")
        robot.coord_frame = base_frame
        robot.coord_frame = robot.target_pose
        robot.move_linear((0, 0, -50, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))
        robot.coord_frame = work_frame
        print("Moving to origin of work frame ...")
        robot.move_linear((0, 0, 0, 0, 0, 0))

        print("Final target pose in work frame: {}".format(robot.target_pose))
        print("Final pose in work frame: {}".format(robot.pose))


if __name__ == '__main__':
    main()

