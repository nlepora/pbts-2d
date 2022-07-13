import os, numpy as np
from turtle import width
import matplotlib, matplotlib.pylab as plt
from matplotlib import animation

from cri.robot import quat2euler, euler2quat, inv_transform

temp_path = os.environ["TEMPPATH"]


class PlotContour:
    def __init__(self, path = temp_path,
                name="contour",  
                frame_rate=10, 
                position=[0, 0, 640, 640],
                poses=[[0,0],[0,0]],
                work_frame = [0, 0, 0, 0, 0, 0],
                r = [0,0], inv = 1, **kwargs):
        self._name = name
        self._path = path
        self._fig = plt.figure(name, figsize=(5, 5))
        self._fig.clear()
        self._ax = self._fig.add_subplot(111) 
        self._ax.set_aspect('equal', adjustable='box')
        
        self._ax.plot(np.array(poses)[:,0], np.array(poses)[:,1], ':w')  

        self.r = r[:2]/np.linalg.norm(r[:2])
        self.a, self.i = (work_frame[5]-90)*np.pi/180, inv 

        mgr = plt.get_current_fig_manager()
        if matplotlib.get_backend()=='TkAgg': mgr.window.wm_geometry('{}x{}'.format(*position[2:]))  
        if matplotlib.get_backend()=='QT5Agg': mgr.window.setGeometry(*position)

        self._writer = animation.writers['ffmpeg'](fps=frame_rate)
        self._writer.setup(self._fig, os.path.join(path, name+'.mp4'), dpi=100)


    def update(self, v, **kwargs):
        v_q = euler2quat([0, 0, 0, 0, 0, self.i*v[-1,5]+self.a], axes='rxyz')
        d_q = euler2quat([*self.r[::-self.i], 0, 0, 0, 0], axes='rxyz')
        d = 5*quat2euler(inv_transform(d_q, v_q), axes='rxyz')

        rv = np.zeros(np.shape(v))
        rv[:,0] = np.cos(self.a)*v[:,0] - np.sin(self.a)*v[:,1]
        rv[:,1] = np.sin(self.a)*v[:,0] + np.cos(self.a)*v[:,1]
        
        self._ax.plot(self.i*rv[-2:,0], rv[-2:,1], '-r') 
        self._ax.plot(self.i*rv[-2:,0]+[d[0],-d[0]], rv[-2:,1]+[d[1],-d[1]], '-b', linewidth=0.5) 

        plt.pause(0.0001)
        self._writer.grab_frame()
        self._fig.show()


    def finish(self, flag=''):
        if flag is 'png': self._fig.savefig(os.path.join(self._path, self._name+'.png'), bbox_inches="tight") 
        self._writer.finish() 


def main():
    None

if __name__ == '__main__':
    main()
