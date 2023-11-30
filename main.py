from AspRatioSystem import AspRatioSystem as ARS
import cv2

img = cv2.imread("./Dataset.jpg", 1) # 4:3 video frame;size=800x600
target_size = (1080, 1920)  # 16:9 video frame;size=1920x1080
Pillarbox = ARS(img=img, t_size=target_size)
Pillarbox.NLS()
Pillarbox.plot_result()
# Pillarbox.Resize()
# Pillarbox.plot_result()
# Pillarbox.Pillarboxing()
# Pillarbox.plot_result()