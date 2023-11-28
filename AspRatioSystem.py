# Author: Fu, Guan Hao | sam27368@gmail.com
# Semtemper, 2023
# This program is to implement how to convert video frame with old video aspect ratio 4:3 to the newer standard of video apect 16:9
# =================================================================================================================================
import cv2
import copy
import numpy as np

class AspRatioSystem():
    def __init__(self, img, t_size) :
        self.input=img 
        self.t_size = t_size
        self.output = None  # converted to 16:9 video frame

    #  Directly resize 4:3 aspect ratio video to a 16:9 TV/monitor
    # input: 4:3 video frame output: resized 16:9 video frame
    def Resize(self):
        img2 = self.input.copy()
        (target_h, target_w) = self.t_size # (height,width) is (1080,1920) on a 16:9 monitor 
        self.output =  cv2.resize(img2, (target_w, target_h), cv2.INTER_AREA)

    # If you have a 4:3 aspect ratio video and display it on a 16:9 TV/monitor in the best way
    # input: 4:3 video frame output: 16:9 video frame
    def Pillarboxing(self):
        img2 = self.input.copy()
        (h, w) = img2.shape[:2]
        (target_h, target_w) = self.t_size      # (height,width) is (1080,1920) on a 16:9 monitor 
                
        target_ratio = target_h/h               # Calcuate the ratio of height 
        converted_w = int(w * target_ratio)     # Applied to width
        img2 = cv2.resize(img2, (converted_w, target_h), cv2.INTER_AREA)
        self.output  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        print(f"img2.shape={img2.shape}, self.output={self.output.shape}")

        center_diff_w = int(target_w/2 - converted_w/2)
        self.output[:, center_diff_w:target_w-center_diff_w] = img2[:] # padding 4:3 video on a 16:9 monitor

    # If you playback a 16:9 (Wide screen) program on a 4:3 monitor
    # input: 16:9 video frame output: 4:3 video frame   
    def Letterboxing(self):
        print("TODO")

    # Nonlinear Scaling Method
    def NLS(self):
        img2 = self.input.copy()
        (h, w) = img2.shape[:2]
        (target_h, target_w) = self.t_size  # (height,width) is (1080,1920) on a 16:9 monitor       

        f1 = target_h/h             # zoom factor of the linear area
        Wm = int(w * f1)                 # the width of the post scaling area M
        Wl_r = int((target_w - Wm)/2)    # the width of the post scaling area L R
        # print(f"f1 = {f1}, Wm={Wm}")
        a1= 0                       # the first element of the equal difference progression
        d = ((1/f1)-a1)/Wl_r   # the common difference on non-Linear region

        img2 = cv2.resize(img2, (Wm, target_h), cv2.INTER_AREA) # Linear area(M)
        self.output  = np.zeros((target_h, target_w, 3), dtype=np.uint8)    # Black area which image size is (1920,1080)
        print(f"img2.shape={img2.shape}, distance between adjacent pixels={d}, the width of the post scaling area L={Wl_r}")

    def plot_result(self):
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.imshow("demo", self.output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()