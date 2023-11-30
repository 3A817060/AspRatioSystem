# Author: Fu, Guan Hao | sam27368@gmail.com
# Semtemper, 2023
# This program is to implement how to convert video frame with old video aspect ratio 4:3 to the newer standard of video apect 16:9
# =================================================================================================================================
import cv2
import copy
import numpy as np
import Bicubic.bicubic as BC
import time

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
        # print(f"img2.shape={img2.shape}, self.output={self.output.shape}")

        center_diff_w = int(target_w/2 - converted_w/2)
        self.output[:, center_diff_w:target_w-center_diff_w] = img2[:] # padding 4:3 video on a 16:9 monitor

    # If you playback a 16:9 (Wide screen) program on a 4:3 monitor
    # input: 16:9 video frame output: 4:3 video frame   
    def Letterboxing(self):
        print("TODO")

    # Nonlinear Scaling Method
    def NLS(self):
        # --------------------------------
        # image input & get the size of input image and target image
        imgc = self.input.copy()
        (h, w) = imgc.shape[:2]
        (target_h, target_w) = self.t_size  # (height,width) is (1080,1920) on a 16:9 monitor       

        # --------------------------------
        # the parameter of NLS function
        Rm = 0.85                                # the ratio of the width of linear area 
        Rl_r = int(w*((1-Rm)/2))                # the ratio of the width of nonlinear area 
        fl = target_h/h                         # upscaling factor of the linear area of Height
        Wm = int(w * Rm * fl)                   # the width of the post scaling area M
        Wl_r = int((target_w - Wm)/2)           # the width of the post scaling area L R
        a1= 0                                   # the first element of the equal difference progression
        # d = ((1/f1)-a1)/Wl_r                    # the common difference on non-Linear region
        d = 1/(fl*Wl_r)                          # the common difference on non-Linear region
        # d *= fl_w
        print(f"f1 = {fl}, Wm={Wm}, Wl_r={Wl_r}")

        # --------------------------------
        # Get 3 seperated region on original image
        # Padding Region L 
        imgL = imgc[:, 0:Rl_r]
        imgL_pa = np.zeros((h+3,Rl_r+3,3))
        imgL_pa[:-3,:,:] = imgc[:, 0:Rl_r+3, :]
        # Padding Region M 
        imgM = imgc[:, Rl_r:w-Rl_r]
        imgM_pa = np.zeros((h+3,w-(2*Rl_r),3))
        imgM_pa[:-3,:,:] = imgc[:, Rl_r:w-Rl_r]
        # Padding Region R
        imgR = imgc[:, w-Rl_r:]
        # imgR = cv2.flip(imgR, 1)
        imgR_pa = np.zeros((h+3,Rl_r+3,3))
        imgR_pa[:-3,:,:] = imgc[:, w-Rl_r-3:]
        print(f"imgc.shape={imgc.shape}, imgM.shape={imgM.shape}, imgR_pa.shape={imgR_pa.shape}, imgL_pa.shape={imgL_pa.shape}, distance={d}")

        # --------------------------------
        # Resize 3 seperated region depend on original image
        start_time = time.time()
        # self.plot_result_1(imgL)
        imgL_r = BC.bicubic(imgL, imgL_pa, d, [target_h, Wl_r], is_reverse=False)                  # Nonlinear area(L)
        # self.plot_result_1(imgL_r)
        # imgR_r = cv2.resize(imgR, (Wl_r, target_h), interpolation=cv2.INTER_CUBIC)    # Nonlinear area(L)

        # imgM_r = BC.bicubic(imgM, imgM_pa, d, [target_h, Wm], False,linear=True)    # Linear area(M)
        imgM_r = cv2.resize(imgM, None, fx=fl, fy=fl, interpolation=cv2.INTER_CUBIC)    # Linear area(M)

        imgR_r = BC.bicubic(imgR, imgR_pa, d, [target_h, Wl_r], is_reverse=True)                  # Nonlinear area(R)
        # imgR_r = BC.bicubic(imgR, imgR_pa, d, [target_h, Wl_r], True)                            # Nonlinear area(R)
        # imgR_r = cv2.resize(imgR, (Wl_r, target_h), interpolation=cv2.INTER_CUBIC)    # Nonlinear area(R)
        end_time = time.time()
        print(f"imgM_resize.shape={imgM_r.shape}, imgR_resize.shape={imgR_r.shape}")
        print(f"elapsed_time={end_time - start_time}")
        # --------------------------------
        # Pad resized image to the output empty frame
        self.output  = np.zeros((target_h, target_w, 3), dtype=np.uint8)    # Black area which image size is (1920,1080)
        center_diff_w = int(target_w/2 - imgM_r.shape[1]/2)
        self.output[:, 0:Wl_r] = imgL_r[:]                                  # padding Nonlinear area(R) on a 16:9 monitor
        self.output[:, center_diff_w:target_w-center_diff_w] = imgM_r[:]    # padding Linear area(M) on a 16:9 monitor
        self.output[:, Wl_r+Wm:] = imgR_r[:]                                  # padding Nonlinear area(L) on a 16:9 monitor
        # print(f"imgc.shape={imgc.shape}, imgM.shape={imgM.shape}, distance between adjacent pixels={d}, the width of the post scaling area L={Wl_r}")

    def plot_result(self):
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.imshow("demo", self.output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_result_1(self, img):
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.imshow("demo", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()