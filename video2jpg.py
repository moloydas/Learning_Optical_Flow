import cv2
import os

# Read the video from specified path 
video_file = "VID_20200911_181140"
cam = cv2.VideoCapture("./videos/" + video_file + ".mp4") 
  
try: 
      
    # creating a folder named data 
    if not os.path.exists("./videos/" + video_file): 
        os.makedirs("./videos/" + video_file) 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
  
# frame 
currentframe = 0
  
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret: 
        # if video is still left continue creating images 
        name = './videos/'+ video_file +'/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
  
        # writing the extracted images 
        cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 