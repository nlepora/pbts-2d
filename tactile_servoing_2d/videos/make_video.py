import os, re

from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips, ImageClip
import moviepy.video.fx.all as vfx

data_path = os.path.join(os.environ["DATAPATH"], "open", "tactile-servoing-2d-dobot")
replay_dir = os.path.join(data_path, "digitac", "servo_surface2d", "disk")
   

def main(replay_dir):
    # combine camera videos
    camera_dir = os.path.join(replay_dir, "camera")
    alphanum = lambda key: [c for c in re.search('camera_(.*?).mp4',key).group(1)]
    integer = lambda key: int(''.join(alphanum(key)))
    camera_files = sorted(os.listdir(camera_dir), key=integer)#[:10]

    # load videos
    video1 = VideoFileClip(os.path.join(replay_dir, "Contour.mp4")).crop(x1=0, y1=0, x2=480, y2=480)
    video2 = VideoFileClip(os.path.join(replay_dir, "Frames.mp4")).crop(x1=0, y1=0, x2=480, y2=480)

    # create new video        
    clips, clips_camera = [[], []]
    for item, camera_file in enumerate(camera_files[1:]):
        print(camera_file)
            
        clip0 = VideoFileClip(os.path.join(camera_dir, camera_file))
        if clip0.fps * clip0.end > 1: 
            clip0.reader.close() # prevents memory issues
            
        clip1 = ImageClip(video1.get_frame(item/video1.fps)).set_duration(clip0.end)
        clip2 = ImageClip(video2.get_frame(item/video2.fps)).set_duration(clip0.end)

        x1=160; y1=0; x2=160+960; y2=720
        clips.append(clips_array([[clip0.crop(x1,y1,x2,y2)], 
                                    [clips_array([[clip1, clip2]])]]))
        clips_camera.append(clip0)

    video = concatenate_videoclips(clips, method='compose')
    video_camera = concatenate_videoclips(clips_camera, method='compose')
    video_small = video.fx(vfx.resize, width=640)
    video_small_fast = video_small.fx(vfx.speedx, factor=2)
        
    #    video.write_videofile(os.path.join(home_dir, replay_dir, file, 'Video.mp4'))    
    #    video_camera.write_videofile(os.path.join(home_dir, replay_dir, file, 'Camera.mp4'))
    #    video_small.write_videofile(os.path.join(home_dir, replay_dir, file, 'Video_small.mp4'))
    video_small_fast.write_videofile(os.path.join(replay_dir, 'Video_small_fast.mp4'))


if __name__ == '__main__':
    main(replay_dir)