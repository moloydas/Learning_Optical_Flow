Test code to learn dense optical flow using horn and schunck method

optical_flow.py: code to run to get optical flow
syn_img.py: generated synthetic images to test optical flow
video2jpg: extracts images from videos

TO DO:
make it faster using GPU
farneback's algorithm

images to video
ffmpeg -r 15 -start_number 1 -i %d_opti_arrow.jpg -vframes 1000 opti_color_arrow.mp4
