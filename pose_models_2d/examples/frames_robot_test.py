from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay, CvPreprocVideoCamera  
from vsp.processor import CameraStreamProcessor, AsyncProcessor


def make_sensor(): # amcap: reset all settings; autoexposure off; saturdation max
    camera = CvPreprocVideoCamera(source=1,
                crop=[320-128-10, 240-128+10, 320+128-10, 240+128+10],
                size=[128, 128], 
                threshold=[61, -5],
                exposure=-6)
    for _ in range(5): camera.read() # Hack - camera transient    
    return AsyncProcessor(CameraStreamProcessor(camera=camera, 
                display=CvVideoDisplay(name='sensor'),
                writer=CvImageOutputFileSeq()))


# Move robot and collect data
with SyncRobot(Controller()) as robot, make_sensor() as sensor:

    robot.linear_speed = 10
    robot.coord_frame = [310, 0, -122, 0, 0, 0] # careful
    robot.move_linear([0, 0, 0, 0, 0, 0])  

    print('\nTest synchronous frames capture...')        
    robot.move_linear([0, 0, -10, 0, 0, 0])
    robot.move_linear([0, 0, -10-5, 0, 0, 0])
    frames_sync = sensor.process(num_frames=5, start_frame=1, outfile=r'C:/Temp/frames_sync.png') 
    robot.move_linear([0, 0, -10, 0, 0, 0])
    print(f'frames.shape={frames_sync.shape}') 

    print('\nTest asynchronous frames capture...')        
    robot.move_linear([0, 0, -10, 0, 0, 0])
    sensor.async_process(num_frames=30, start_frame=1, outfile=r'C:/Temp/frames_async.png') 
    robot.move_linear([0, 0, -10-5, 0, 0, 0])
    robot.move_linear([0, 0, -10, 0, 0, 0])
    frames_async = sensor.async_result()
    print(f'frames.shape={frames_async.shape}')      

    robot.move_linear([0, 0, 0, 0, 0, 0])


# display results        
import matplotlib.pylab as plt # needs to be after initialising controller (strange bug)
for i, frame in enumerate(frames_async):
    plt.title(f'Frame {i+1}')
    plt.imshow(frame, cmap="Greys")
    plt.xticks([]); plt.yticks([]); 
    plt.pause(0.1)
plt.show()
