import cv2 as cv
import numpy as np

class CameraTrajectory:
    """
    Computes the trajectory of a camera, based on feature tracks
    """
    def __init__(self, feature_tracks, smoothing_rad):
        self.feature_tracks = feature_tracks     
        self.window_rad = smoothing_rad

    def find_trajectory(self):
        lastT = np.array([[1, 0, 0],
                          [0, 1, 0]]) #default transform is no change
        
        dtraj = [] #change in pose per frame
        
        
        for tracks, frame_idx in self.feature_tracks:    #Unlikely bug: a frame with zero tracks               
            if frame_idx == 0: #skip first frame
                continue                                    
            
            if len(tracks) > 0: #if no tracks, use last T again
                prev_coord = np.int64(tracks[:, -2])
                curr_coord = np.int64(tracks[:, -1])

                T, _ = cv.estimateAffinePartial2D(prev_coord, curr_coord, method=cv.LMEDS)
                if T is None:
                    T = lastT    

            dx = T[0, 2]
            dy = T[1, 2]
            da = np.arctan2(T[1, 0], T[0, 0])

            dtraj.append(np.array([dx, dy, da]))
            lastT = T
        
        self.dtraj = np.array(dtraj)
        self.traj = np.cumsum(dtraj, axis=0)
    
    def smooth(self):
        """
        moving average smoothing of trajectory
        """
        window_width = 2*self.window_rad + 1
        self.smooth_traj = np.zeros_like(self.traj)

        kernel = np.ones(window_width)/window_width

        for i in range(3):
            padded_traj = np.pad(self.traj[:, i], self.window_rad, 'edge')
            self.smooth_traj[:, i] = np.convolve(padded_traj, kernel, mode='valid')

        self.smooth_dtraj = self.dtraj + self.smooth_traj - self.traj
       

    def no_movement(self):
        self.smooth_traj = np.zeros_like(self.traj)
        self.smooth_dtraj = self.dtraj + self.smooth_traj - self.traj

    def find_and_smooth(self, mode = "smooth"):
        """
        Compute and smooth trajectory
        @param mode, ["smooth", "None"], default "smooth"
        """
        self.find_trajectory()
        if mode == "None": #only for videos without moving camera
            self.no_movement()
        else:
            self.smooth()

        return self.traj, self.dtraj, self.smooth_traj, self.smooth_dtraj
    
    