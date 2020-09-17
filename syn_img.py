import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.ones((200,200)) * 255
img = cv2.rectangle(img, (50,50), (100,100), (0,0,0), -1)
cv2.imwrite("./test_images/img_1.jpg", img)

img = np.ones((200,200)) * 255
img = cv2.rectangle(img, (52,52), (102,102), (0,0,0), -1)
cv2.imwrite("./test_images/img_2.jpg", img)

img = np.ones((200,200)) * 255
img = cv2.rectangle(img, (50,50), (100,100), (0,0,0), -1)
cv2.imwrite("./test_images/img_1_large_disp.jpg", img)

img = np.ones((200,200)) * 255
img = cv2.rectangle(img, (55,55), (105 ,105), (0,0,0), -1)
cv2.imwrite("./test_images/img_2_large_disp.jpg", img)
