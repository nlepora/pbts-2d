import os, json
from pathlib import Path, PureWindowsPath # for Unix compatibility

from cri.robot import SyncRobot
# from cri.controller import Mg400Controller as Controller
from cri.controller import DummyController as Controller
from vsp.sensor_replay import SensorReplay as Sensor
# from vsp.sensor_dummy import SensorDummy as Sensor

from tactile_servoing_2d.lib.common import home, control
from tactile_servoing_2d.lib.visualise.plot_contour2d import PlotContour
from tactile_servoing_2d.lib.visualise.plot_frames import PlotFrames

from pose_models_2d.lib.models.cnn_model import CNNmodel as Model
# from pose_models_2d.lib.models.dummy_model import DummyModel as Model

data_path = os.path.join(os.environ["DATAPATH"], "open", "tactile-servoing-2d-dobot")
replay_dir = os.path.join(data_path, "digit", "servo_edge2d", "disk")


# Specify directories and files
def main(replay_dir):

    # Load/modify meta data
    with open(os.path.join(replay_dir, "meta.json"), 'r') as f:
        meta = json.load(f)   
    # meta["linear_speed"] = 100

    # Absolute posix paths
    for key in [k for k in meta.keys() if "file" in k or "dir" in k]:
        meta[key] = os.path.join(data_path, meta[key])
        meta[key] = Path(PureWindowsPath(meta[key])).as_posix() # for Unix

    # Startup/load model
    model = Model()
    model.load_model(**meta)

    # Initialize plots
    figs = [PlotContour(**meta), PlotFrames()] # argument replay_dir to save

    # Control robot
    with SyncRobot(Controller()) as robot, Sensor(**meta) as sensor:       
        home(robot, **meta)     
        pred, _ = control(robot, sensor, model, figures=figs, **meta)    

    # Save run to file
    # pred.to_csv(meta["test_df_file"], index=False)


if __name__ == '__main__':
    main(replay_dir)
    