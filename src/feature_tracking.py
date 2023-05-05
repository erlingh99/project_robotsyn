import numpy as np
import cv2 as cv


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.01,
                       minDistance = 30, #big to spread all over image
                       blockSize = 3)


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA) #black text background
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA) #white foreground


class FeatureTracker:
    def __init__(self, cam, playback_speed = 'Fast'):
        """
        @param cam, video object to read frames from
        @param playback_speed = ['Fast', 'Realistic', 'None'], default 'Fast'
        """
        self.track_len = 10
        self.detect_interval = 5
        self.cam = cam

        self.N_frames = int(self.cam.get(cv.CAP_PROP_FRAME_COUNT))

        if playback_speed == "Realistic":
            self.delay = int(1000/self.cam.get(cv.CAP_PROP_FPS))
        elif playback_speed == 'Fast':
            self.delay = 1
        else:
            self.delay = -1

        self.im_height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        

    def run(self):
        all_tracks = []
        tracks = []
        frame_idx = 0

        while frame_idx < self.N_frames:
            _ret, frame = self.cam.read()
            assert _ret, f"bad frame, {frame_idx}/{self.N_frames}"
            
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if self.delay != -1:
                vis = frame.copy() #frame to show tracking on
            
            if len(tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2) #pos of features in prev frame
                p1, _, _ = cv.calcOpticalFlowPyrLK(img0, img1, p0, None)#, **lk_params) #pos of features in current frame
                p0r, _, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, None)#, **lk_params) #reverse track of features in prev frame for robustness
                d = abs(p0-p0r).reshape(-1, 2).max(-1) #max error between back guess and original
                good = d < 1 #keep tracks with small error, in pixels
                
                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good): #append newest pos for all tracks, terminate lost tracks                   
                    if not good_flag: #track no longer found, dont include
                        continue
                    tr.append((x, y)) #add new track point

                    if len(tr) > self.track_len: #delete tracks older than track_len frames
                        continue              
                        tr = tr[1:] #could prune, but then n-tracks will still grow

                    new_tracks.append(tr)
                                                            
                tracks = new_tracks                

                
                #add tracks, lines and text to show frame
                if self.delay != -1:
                    for tr in tracks:
                        cv.circle(vis, np.int32(tr[-1]), 2, (0, 255, 0), -1)                    
                    cv.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                    draw_str(vis, (20, 20), f"track count: {len(tracks)}")
                    draw_str(vis, (20, self.im_height - 20), f"Frame: {frame_idx + 1}/{self.N_frames}")

            all_tracks.append((np.array([np.array(tr[-2:]) for tr in tracks]), frame_idx)) #save the last 2 data-points of each track, and tag with what frame

            if frame_idx % self.detect_interval == 0: #find new features at every detect_interval frames
                mask = np.ones_like(frame_gray)*255 #where to look for features in frame
                
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1) #do not look for features in small circle around where features already are

                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])


            frame_idx += 1
            prev_gray = frame_gray
            if self.delay != -1:
                cv.imshow('lk_track', vis)
                cv.waitKey(self.delay)

        cv.destroyAllWindows()       
        return all_tracks