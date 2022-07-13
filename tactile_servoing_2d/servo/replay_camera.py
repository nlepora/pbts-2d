import os, sys, json
from pathlib import Path, PureWindowsPath # for Unix compatibility

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
# from cri.controller import DummyController as Controller
# from vsp.sensor_replay import SensorReplay as Sensor
from vsp.sensor_dummy import SensorDummy as Sensor

from tactile_servoing_2d.lib.common_camera import home, control
# from pose_models_2d.lib.models.cnn_model import CNNmodel as Model
from pose_models_2d.lib.models.dummy_model import DummyModel as Model

from vsp.processor import CameraStreamProcessor, AsyncProcessor
from vsp.video_stream import CvVideoCamera, CvVideoDisplay, CvVideoOutputFile  

def Camera():
    return AsyncProcessor(CameraStreamProcessor(
            camera=CvVideoCamera(frame_size=(1280,720), source=0),
            display=CvVideoDisplay(name='camera'),
            writer=CvVideoOutputFile(frame_size=(1280,720)),
            )) 


data_path = os.path.join(os.environ["DATAPATH"], "open", "tactile-servoing-2d-dobot")
replay_dir = os.path.join(data_path, "tactip-127", "servo_surface2d", "flower")

# Load/modify meta data
with open(os.path.join(replay_dir, "meta.json"), 'r') as f:
    meta = json.load(f)   
meta["linear_speed"] = 100
meta["test_camera_dir"] = meta["test_image_dir"].replace("frames_bw", "camera")

# Absolute posix paths
for key in [k for k in meta.keys() if "file" in k or "dir" in k]:
    meta[key] = os.path.join(data_path, meta[key])
    meta[key] = Path(PureWindowsPath(meta[key])).as_posix() # for Unix

# Startup/load model
model = Model()
model.load_model(**meta)

# Control robot
with SyncRobot(Controller()) as robot, Sensor(**meta) as sensor, Camera() as camera:       
    home(robot, **meta)     
    control(robot, sensor, model, camera, **meta)    
    