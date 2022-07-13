import os, numpy as np
import matplotlib, matplotlib.pylab as plt
from matplotlib import animation

temp_path = os.environ["TEMPPATH"]


class PlotFrames:
    def __init__(self, path = temp_path,
                name="frames", 
                frame_rate=10, 
                position=[0, 0, 480, 480]):
        self._name = name
        self._path = path
        self._fig = plt.figure(name, figsize=(5, 5))
        self._ax = self._fig.add_subplot(111)

        mgr = plt.get_current_fig_manager()
        if matplotlib.get_backend()=='TkAgg': mgr.window.wm_geometry("{}x{}".format(*position[2:]))  
        if matplotlib.get_backend()=='QT5Agg': mgr.window.setGeometry(*position)

        self._writer = animation.writers['ffmpeg'](fps=frame_rate)
        self._writer.setup(self._fig, os.path.join(path, name+'.mp4'), dpi=100)


    def update(self, frames, **kwargs):
        plt.cla()
        for frame in frames:
            self._ax.imshow(1 - np.squeeze(np.transpose(frame)), cmap="Greys")
            self._ax.set_xticks([]); self._ax.set_yticks([])

            plt.pause(0.0001)
            self._writer.grab_frame()
        self._fig.show()


    def finish(self, flag=''):
        if flag is 'png': self._fig.savefig(os.path.join(self._path, self._name+'.png'), bbox_inches="tight") 
        self._writer.finish() 


def main():
    pass

if __name__ == "__main__":
    main()
