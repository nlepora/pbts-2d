import os, sys, numpy as np, pandas as pd
from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
# from cri.controller import DummyController as Controller
from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay, CvPreprocVideoCamera   
from vsp.processor import CameraStreamProcessor, AsyncProcessor
# from vsp.sensor_replay import SensorReplay as Sensor

from pose_models_2d.lib.models.cnn_model import CNNmodel as Model

def make_sensor(): # amcap: reset all settings; autoexposure off; saturdation max
    camera = CvPreprocVideoCamera(source=1,
                # crop=[320-128-2, 240-128+20, 320+128-2, 240+128+20],
                crop=[0, 0, 640, 480],
                exposure=-7, 
                size=[120, 160], # digitac requires dimensions reversing
                # size=[128, 128], # ensure consistent with meta data
                threshold=[61, -5])
    for _ in range(5): camera.read() # Hack - camera transient    
    return AsyncProcessor(CameraStreamProcessor(camera=camera, 
                display=CvVideoDisplay(name='sensor'),
                writer=CvImageOutputFileSeq()))


temp_dir = os.environ['TEMPPATH']

model_dir = os.path.join(os.environ["DATAPATH"], "dobot", "digitac", "edge2d", "train", "train2d_cnn")
work_frame = [290, 0, -112, 0, 0, -90-8]

# model_dir = os.path.join(os.environ["DATAPATH"], "dobot", "tactip-331", "edge2d", "train", "train2d_cnn")
# work_frame = [288, 0, -100, 0, 0, -90]

model = Model()
model.load_model(os.path.join(model_dir, "model.h5"), )

with SyncRobot(Controller()) as robot, make_sensor() as sensor:     
    
    robot.coord_frame = work_frame

    pred, var = [[[], []] for _ in range(2)]

    robot.move_linear([0,]*6)

    for i, x in enumerate(np.linspace(5,-5,11)):
        robot.move_linear([0, x, -5, 0, 0, 0])
        frames = sensor.process(num_frames=1, start_frame=1, outfile=fr"{temp_dir}\frames_x_{i}.png") 
        pred[0].append(model.predict_from_saved(temp_dir, f"frames_x_{i}.png")[0])           
        # pred[0].append(model.predict(frames/255)[0])
        var[0].append(x) 
        print(f'{x}: prediction={pred[0][-1]}') 

    robot.move_linear([0,]*6)

    for i, th in enumerate(np.linspace(45,-45,11)):
        robot.move_linear([0, 0, -5, 0, 0, th])
        frames = sensor.process(num_frames=1, start_frame=1, outfile=fr"{temp_dir}\frames_th_{i}.png") 
        pred[1].append(model.predict_from_saved(temp_dir, f"frames_th_{i}.png")[1])           
        # pred[1].append(model.predict(frames/255)[1]) 
        var[1].append(th) 
        print(f'{th}: prediction={pred[1][-1]}') 

    robot.move_linear([0,]*6)

# display results  
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=len(pred), figsize=(len(pred)*5,5))
for i, ax in enumerate(axes): 
    ax.plot(var[i], pred[i], '-b.')
    ax.plot(var[i], var[i], ':k')
    ax.grid(True)
plt.show()