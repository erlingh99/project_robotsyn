"""
Script for smoothing video
Depth is not handled well, so videos with big depth differences will suffer from parallax.
Ideally all tracked features are at the same depth, or all are far away, however no actions are taken to guarantee this.
"""
import skvideo
import skvideo.io

from feature_tracking import FeatureTracker
from trajectory import CameraTrajectory
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

from os.path import basename

print(__doc__)



vid_src = "../videos/lop.mp4"
cam = cv.VideoCapture(vid_src)
w, h = int(cam.get(cv.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
FPS = cam.get(cv.CAP_PROP_FPS)

## Find feature tracks
feature_tracks = FeatureTracker(cam, 'Realistic').run()        
print("Feature tracks created")
## Compute camera movement, and stabilized camera movement
traj, dtraj, smooth_traj, smooth_dtraj = CameraTrajectory(feature_tracks, 30).find_and_smooth()

## Compute stable camera transforms
smooth_Ts = []
for dx, dy, da in smooth_dtraj:  
    T = np.array([[np.cos(da), -np.sin(da), dx],
                  [np.sin(da),  np.cos(da), dy]])            
    smooth_Ts.append(T)



video_writer = skvideo.io.FFmpegWriter("../videos/out_" + basename(vid_src), inputdict={'-r':str(FPS)})
## Warp images and show
cam.set(cv.CAP_PROP_POS_FRAMES, 0)
for i, sT in enumerate(smooth_Ts):
    _, frame = cam.read()
    stab_frame = cv.warpAffine(frame, sT, (w, h))

    both = cv.hconcat([frame, stab_frame])    

    cv.imshow('Comparison', both)
    both = cv.cvtColor(both, cv.COLOR_BGR2RGB)
    video_writer.writeFrame(both)
    c = cv.waitKey(int(1000/FPS))
    if c == 27: #ESC_key
        break

cv.destroyAllWindows()    
video_writer.close()


fig, axs = plt.subplots(1, 3)
for i in range(3):
    axs[i].plot(traj[:, i])
    axs[i].plot(smooth_traj[:, i])    
    axs[i].set_title(["x","y","a"][i])

fig.legend(["traj", "smooth"])
plt.suptitle("Trajectories")
plt.show()