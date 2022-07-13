import os

from tactile_servoing_2d.servo.replay import main as main_replay
from tactile_servoing_2d.videos.make_video import main as main_video


data_path = os.path.join(os.environ["DATAPATH"], "open", "tactile-servoing-2d-dobot")

sensors = ["digit", "digitac", "tactip-127", "tactip-331"]
shapes = ["disk", "flower", "square"]
tasks = ["servo_edge2d", "servo_surface2d"]

for sensor in sensors:
    for shape in shapes:
        for task in tasks:
            replay_dir = os.path.join(data_path, sensor, task, shape)
            try:
                main_replay(replay_dir)
                main_video(replay_dir)
            except:
                pass
            