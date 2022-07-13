import os, numpy as np
from copy import copy

from cri.robot import quat2euler, euler2quat, inv_transform

np.set_printoptions(precision=2, suppress=True)


def home(robot, home_pose, linear_speed, angular_speed, robot_tcp=[0,]*6, base_frame=[0,]*6, **kwargs):
    robot.tcp = robot_tcp
    robot.linear_speed = linear_speed
    robot.angular_speed = angular_speed
    robot.coord_frame = base_frame
    robot.move_linear(home_pose)


def control(robot, sensor, model, camera, 
            test_image_dir, test_camera_dir, target_names, num_frames, work_frame,       
            num_steps, r, kp, ki, ei_bnd, 
            **kwargs):
    y, e, ei, u, v = [np.zeros((num_steps+1, 6)) for _ in range(5)]
    u[0,], v[0,], y[0,] = [copy(r) for _ in range(3)]
    ei[0,] = [0,]*6

    pose_, pred_ = [[f"{v}_{_+1}" for _ in range(6)] for v in ["pose", "pred"]]
    target_inds = [pose_.index(t) for t in target_names]

    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_camera_dir, exist_ok=True)

    # start in safe position
    robot.coord_frame = work_frame
    robot.move_linear([0, 0, -50, 0, 0, 0])   

    # step method
    for i in range(num_steps):
        frames_file = f"frame_{i+1}"
        camera_file = f"camera_{i+1}"
            
        # control signal
        r_q = euler2quat(r, axes='rxyz')
        y_q = euler2quat(-y[i,], axes='rxyz')           
        e[i+1,] = quat2euler(inv_transform(r_q, y_q), axes='rxyz')
        ei[i+1,] = np.minimum(np.maximum( e[i+1,]+ei[i,], ei_bnd[1]), ei_bnd[0] )
        u[i+1,] = kp*e[i+1,] + ki*ei[i+1,] 

        # transform control signal to sensor frame                 
        u_q = euler2quat(u[i+1,], axes='rxyz')
        v_q = euler2quat(v[i,], axes='rxyz')
        v[i+1,] = quat2euler(inv_transform(u_q, v_q), axes='rxyz')

        # move to new pose, grab frames, and make prediction
        camera.async_process(outfile=os.path.join(test_camera_dir, camera_file+".mp4"))
        robot.move_linear(v[i+1,])
        camera.async_cancel()
        camera.async_result()
        
        robot.move_linear(v[i+1,])
        frames = sensor.process(num_frames, start_frame=1, outfile=os.path.join(test_image_dir, frames_file+".png"))
        y[i+1,target_inds] = model.predict(frames/255)

        # report
        print(f'{i+1}: v={v[i+1,]}\n     y={y[i+1,]}'.replace('. ',''))
        print(f'joints={np.asarray(robot.joint_angles)}'.replace('. ','')) 

    # finish in safe position
    robot.coord_frame = robot.commanded_pose
    robot.move_linear([0, 0, -30, 0, 0, 0])      


def main():
    None


if __name__ == '__main__':
    None
