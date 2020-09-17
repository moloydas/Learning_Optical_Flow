# optical_flow_LK_pyramids

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parameters
lamda = 10
working_dir = "./videos/VID_20200911_181140/"
file_name = working_dir + "frame"
file_start_no = 1
file_end_no = 212

###
# Funtion to resize image keeping the aspect ratio same
# Credits: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
##
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA, ratio = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None and ratio is None:
        return image, 1
    elif width is None and height is None and ratio is not None:
        dim = (int(w*ratio), int(h*ratio))
        resized = cv2.resize(image, dim, interpolation = inter)
        return resized, ratio

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized, r

for file_no in range(file_start_no,file_end_no):
    img_1_color = cv2.imread(file_name+str(file_no)+'.jpg')
    img_1 = cv2.imread(file_name+str(file_no)+'.jpg',cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(file_name+str(file_no+1)+'.jpg',cv2.IMREAD_GRAYSCALE)

    print("processing: "+ file_name+str(file_no)+'.jpg' + " and " + file_name+str(file_no+1)+'.jpg')

    # Load Test Images
    # img_1 = cv2.imread('./test_images/img_1_large_disp.jpg', cv2.IMREAD_GRAYSCALE)
    # img_2 = cv2.imread('./test_images/img_2_large_disp.jpg', cv2.IMREAD_GRAYSCALE)
    # img_1_color = np.zeros((img_1.shape[0], img_1.shape[1], 3))
    # img_1_color[:,:,0] = img_1[:,:]
    # img_1_color[:,:,1] = img_1[:,:]
    # img_1_color[:,:,2] = img_1[:,:]

    # Rescale input images to reduce computation time
    img_1,scale = image_resize(img_1, height=200)
    img_2,scale = image_resize(img_2, height=200)
    # print("img size: " + str(img_1_color.shape[:2]) + " input scale: " + str(scale))

    fx_mask = np.array([[-1,1],[-1,1]])
    fy_mask = np.array([[-1,-1],[1,1]])
    ft_mask_1 = np.array([[-1,-1],[-1,-1]])
    ft_mask_2 = np.array([[1,1],[1,1]])

    # Create a 4 layer pyramids
    layers = 4
    sigma = 1.04

    u = np.zeros(img_1.shape)
    v = np.zeros(img_1.shape)

    for l in reversed(range(layers)):
        img_1_l = img_1.copy()
        img_2_l = img_2.copy()

        for i in range(l):
            img_1_l,r = image_resize(cv2.GaussianBlur(img_1_l, (5,5), sigma), ratio=0.5)
            img_2_l,r = image_resize(cv2.GaussianBlur(img_2_l, (5,5), sigma), ratio=0.5)

        fx = np.zeros(img_1_l.shape)
        fy = np.zeros(img_2_l.shape)
        ft = np.zeros(img_2_l.shape)

        u = cv2.resize(u*2, (img_1_l.shape[1], img_1_l.shape[0]), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v*2, (img_1_l.shape[1], img_1_l.shape[0]), interpolation=cv2.INTER_LINEAR)

        ## fx fy ft derivatives
        for i in range(0,img_1_l.shape[0]-1):
            for j in range(0,img_1_l.shape[1]-1):
                fx[i,j] = np.average(img_1_l[i:i+2,j:j+2] * fx_mask + img_2_l[i:i+2,j:j+2] * fx_mask)
                fy[i,j] = np.average(img_1_l[i:i+2,j:j+2] * fy_mask + img_2_l[i:i+2,j:j+2] * fy_mask)
                if int(round(i+v[i,j])) < img_1_l.shape[0]-2 and int(round(j+u[i,j])) < img_1_l.shape[1]-2 and int(round(j+u[i,j])) > 0 and int(round(i+v[i,j])) > 0:
                    ft[i,j] = np.average(img_1_l[i:i+2,j:j+2] * ft_mask_1 + img_2_l[int(round(i+v[i,j])):int(round(i+v[i,j]+2)), int(round(j+u[i,j])):int(round(j+u[i,j]+2))] * ft_mask_2)

        ## Calculate u and v
        for i in range(1,fx.shape[0]-1,3):
            for j in range(1,fx.shape[1]-1,3):
                fx_kernel = fx[i-1:i+2, j-1:j+2]
                fy_kernel = fy[i-1:i+2, j-1:j+2]
                ft_kernel = ft[i-1:i+2, j-1:j+2]
                u_n = -np.sum(fy_kernel**2) * np.sum(fx_kernel*ft_kernel) + np.sum(fx_kernel*fy_kernel) * np.sum(fy_kernel*ft_kernel)
                v_n = np.sum(fx_kernel*ft_kernel) * np.sum(fx_kernel*fy_kernel) - np.sum(fx_kernel**2) * np.sum(fy_kernel*ft_kernel)
                d = np.sum(fx_kernel**2) * np.sum(fy_kernel**2) - (np.sum(fx_kernel*fy_kernel))**2

                if d != 0 and abs(int(u_n/d)) < img_1_l.shape[0]/4 and abs(int(v_n/d)) < img_1_l.shape[1]/4:
                    u[i-1:i+2, j-1:j+2] = u[i-1:i+2, j-1:j+2] + np.ones((3,3)) * u_n/d
                    v[i-1:i+2, j-1:j+2] = v[i-1:i+2, j-1:j+2] + np.ones((3,3)) * v_n/d

    opt_flow_mag = np.sqrt(u**2 + v**2)

    hsv = np.zeros((img_1.shape[0],img_1.shape[1],3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = 255

    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    for i in range(0,img_1.shape[0],3):
        for j in range(0,img_1.shape[1],3):
            if opt_flow_mag[i,j] > 1:
                cv2.arrowedLine(img_1_color, 
                                (int(j/scale), int(i/scale)), 
                                (int(j/scale + 3*u[i,j]), int(i/scale + 3*v[i,j])), 
                                color=tuple(bgr[i,j,:].tolist()), 
                                thickness=1, 
                                tipLength=0.2)

    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite("output/opti_mag/" + str(file_no) +'_opti_mag.jpg', opt_flow_mag)
    cv2.imwrite("output/opti_color/" + str(file_no) +'_opti_color.jpg', bgr)
    cv2.imwrite("output/opti_arrow/" + str(file_no) +'_opti_arrow.jpg', img_1_color)

    # cv2.namedWindow('img_1', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('opt_flow_mag', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('opt_flow_color', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('opti_arrow', cv2.WINDOW_NORMAL)

    # cv2.imshow('img_1', img_1)
    # cv2.imshow('opt_flow_mag', opt_flow_mag)
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('opt_flow_color', bgr)
    # cv2.imshow('opti_arrow', img_1_color)
    # # plt.imshow(opt_flow_mag)
    # # plt.show()
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
