import os
import numpy as np
import pandas as pd

np.set_printoptions(precision=2, suppress=True)


def make_target_df(target_df_file, poses_rng, num_poses, num_frames=1, obj_poses=[[0,]*6], moves_rng=[[0,]*6,]*2, **kwargs):
    np.random.seed(0) # make predictable
    poses = np.random.uniform(low=poses_rng[0], high=poses_rng[1], size=(num_poses, 6))
    poses = poses[np.lexsort((poses[:,1], poses[:,5]))]
    moves = np.random.uniform(low=moves_rng[0], high=moves_rng[1], size=(num_poses, 6))

    pose_ = [f"pose_{_+1}" for _ in range(6)]
    move_ = [f"move_{_+1}" for _ in range(6)]
    target_df = pd.DataFrame(columns=["image_name", "data_name", "obj_id", "pose_id", *pose_, *move_])

    for i in range(num_poses * len(obj_poses)):
        data_name = f"frame_{i}"
        i_pose, i_obj = (int(i%num_poses), int(i/num_poses))
        pose = poses[i_pose,:] + obj_poses[i_obj]
        move = moves[i_pose,:]        
        for f in range(num_frames):
            frame_name = f"frame_{i}_{f}.png"
            target_df.loc[-1] = np.hstack((frame_name, data_name, i_obj+1, i_pose+1, pose, move))
            target_df.index += 1

    target_df.to_csv(target_df_file, index=False)
    return target_df


def home(robot, home_pose, linear_speed, angular_speed, robot_tcp=[0,]*6, **kwargs):
    robot.tcp = robot_tcp
    robot.linear_speed, robot.angular_speed = [linear_speed, angular_speed]
    robot.coord_frame = [0,]*6
    robot.move_linear(home_pose)


def collect(robot, sensor, target_df, image_dir, work_frame, tap_move, num_frames, **kwargs):   
    robot.coord_frame = work_frame

    os.makedirs(image_dir, exist_ok=True)
    sensor.process(num_frames=1, start_frame=1, outfile=os.path.join(image_dir, "init.png"))  

    for _, row in target_df[::num_frames].iterrows():
        i_obj, i_pose = (int(row.loc["obj_id"]), int(row.loc["pose_id"]))
        pose = row.loc['pose_1' : 'pose_6'].values.astype(np.float)
        move = row.loc['move_1' : 'move_6'].values.astype(np.float)
        data_name = os.path.join(image_dir, row.loc["data_name"]+'.png')           
        
        print(f"Collecting data for object {i_obj}, pose {i_pose}: ...")
        print(f"pose = {pose}".replace('. ',''))
        print(f"joints={np.asarray(robot.joint_angles)}".replace('. ',''))
        
        robot.move_linear(pose - move)
        robot.move_linear(pose - move + tap_move)
        robot.move_linear(pose + tap_move)
        sensor.process(num_frames, start_frame=1, outfile=data_name)
        robot.move_linear(pose)
      
                   
if __name__ == '__main__':
    None
