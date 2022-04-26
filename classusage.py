import cv2
import matplotlib.pyplot as plt
from SimpleHRNet import SimpleHRNet

model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")
image = cv2.imread("/home/user/Downloads/simple-HRNet-master/demo1.mp4", cv2.IMREAD_COLOR) #这里的文件路径和文件名根据自己要识别的图片来
joints = model.predict(image)
print(joints)

