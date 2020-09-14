import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.ones((200,200)) * 255

img = cv2.rectangle(img, (50,50), (100,100), (0,0,0), -1)

plt.imshow(img)
plt.savefig("img_1.jpg")

img = np.ones((200,200)) * 255
img = cv2.rectangle(img, (52,52), (102,102), (0,0,0), -1)

plt.imshow(img)
plt.savefig("img_2.jpg")
