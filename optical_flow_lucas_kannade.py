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
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image, 1

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
    # img_1_color = cv2.imread(file_name+str(file_no)+'.jpg')
    # img_1 = cv2.imread(file_name+str(file_no)+'.jpg',cv2.IMREAD_GRAYSCALE)
    # img_2 = cv2.imread(file_name+str(file_no+1)+'.jpg',cv2.IMREAD_GRAYSCALE)

    # print("processing: "+ file_name+str(file_no)+'.jpg' + " and " + file_name+str(file_no+1)+'.jpg')

    # Load Test Images
    img_1 = cv2.imread('./test_images/img_1.jpg',cv2.IMREAD_GRAYSCALE)
    img_1_color = cv2.imread('./test_images/img_1.jpg', cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread('./test_images/img_2.jpg',cv2.IMREAD_GRAYSCALE)

    img_1,scale = image_resize(img_1, height=200)
    img_2,scale = image_resize(img_2, height=200)
    print("img size: " + str(img_1_color.shape[:2]) + " input scale: " + str(scale))

    fx = np.zeros(img_1.shape)
    fy = np.zeros(img_2.shape)
    ft = np.zeros(img_2.shape)

    fx_mask = np.array([[-1,1],[-1,1]])
    fy_mask = np.array([[-1,-1],[1,1]])
    ft_mask_1 = np.array([[-1,-1],[-1,-1]])
    ft_mask_2 = np.array([[1,1],[1,1]])

    ## fx fy ft derivatives
    for i in range(0,img_1.shape[0]-1):
        for j in range(0,img_1.shape[1]-1):
            fx[i,j] = np.average(img_1[i:i+2,j:j+2] * fx_mask + img_2[i:i+2,j:j+2] * fx_mask)
            fy[i,j] = np.average(img_1[i:i+2,j:j+2] * fy_mask + img_2[i:i+2,j:j+2] * fy_mask)
            ft[i,j] = np.average(img_1[i:i+2,j:j+2] * ft_mask_1 + img_2[i:i+2,j:j+2] * ft_mask_2)

    u = np.zeros(fx.shape)
    v = np.zeros(fx.shape)

    opt_flow_mag = np.zeros(fx.shape)

    for i in range(1,fx.shape[0]-1,3):
        for j in range(1,fx.shape[1]-1,3):
            fx_kernel = fx[i-1:i+2, j-1:j+2]
            fy_kernel = fy[i-1:i+2, j-1:j+2]
            ft_kernel = ft[i-1:i+2, j-1:j+2]
            u_n = -np.sum(fy_kernel**2) * np.sum(fx_kernel*ft_kernel) + np.sum(fx_kernel*fy_kernel) * np.sum(fy_kernel*ft_kernel)
            v_n = np.sum(fx_kernel*ft_kernel) * np.sum(fx_kernel*fy_kernel) - np.sum(fx_kernel**2) * np.sum(fy_kernel*ft_kernel)
            d = np.sum(fx_kernel**2) * np.sum(fy_kernel**2) - (np.sum(fx_kernel*fy_kernel))**2
            if d != 0:
                u[i-1:i+2, j-1:j+2] = np.ones((3,3)) * u_n/d
                v[i-1:i+2, j-1:j+2] = np.ones((3,3)) * v_n/d
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

    # cv2.imwrite("output/opti_mag/" + str(file_no) +'_opti_mag.jpg', opt_flow_mag)
    # cv2.imwrite("output/opti_arrow/" + str(file_no) +'_opti_arrow.jpg', img_1_color)

    cv2.namedWindow('img_1', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('opt_flow_mag', cv2.WINDOW_NORMAL)
    cv2.namedWindow('opti_arrow', cv2.WINDOW_NORMAL)

    cv2.imshow('img_1', img_1)
    # cv2.imshow('opt_flow_mag', opt_flow_mag)
    cv2.imshow('opti_arrow', img_1_color)
    plt.imshow(opt_flow_mag)
    plt.show()
    cv2.waitKey(0)

    cv2.destroyAllWindows()
