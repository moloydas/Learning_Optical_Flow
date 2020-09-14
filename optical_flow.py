import cv2
import numpy as np
import matplotlib.pyplot as plt

lamda = 15

file_name = "./videos/VID_20200911_181140/frame"

file_str_no = 1

for file_str_no in range(1,327):
    img_1 = cv2.imread(file_name+str(file_str_no)+'.jpg',cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(file_name+str(file_str_no+1)+'.jpg',cv2.IMREAD_GRAYSCALE)

    print("processing: "+ file_name+str(file_str_no)+'.jpg' + " and " + file_name+str(file_str_no+1)+'.jpg')

    # img_1 = cv2.imread('./img_1.jpg',cv2.IMREAD_GRAYSCALE)
    # img_2 = cv2.imread('./img_2.jpg',cv2.IMREAD_GRAYSCALE)

    img_1_output = cv2.resize(cv2.cvtColor(img_1,cv2.COLOR_GRAY2RGB), (800, 800))
    img_1 = cv2.resize(img_1, (200,200))
    img_2 = cv2.resize(img_2, (200,200))

    fx = np.zeros(img_1.shape)
    fy = np.zeros(img_2.shape)
    ft = np.zeros(img_2.shape)

    fx_mask = np.array([[-1,1],[-1,1]])
    fy_mask = np.array([[-1,-1],[1,1]])
    ft_mask_1 = np.array([[-1,-1],[-1,-1]])
    ft_mask_2 = np.array([[1,1],[1,1]])

    ## fx fy ft
    for i in range(0,img_1.shape[0]-1):
        for j in range(0,img_1.shape[1]-1):
            fx[i,j] = np.average(img_1[i:i+2,j:j+2] * fx_mask + img_2[i:i+2,j:j+2] * fx_mask)
            fy[i,j] = np.average(img_1[i:i+2,j:j+2] * fy_mask + img_2[i:i+2,j:j+2] * fy_mask)
            ft[i,j] = np.average(img_1[i:i+2,j:j+2] * ft_mask_1 + img_2[i:i+2,j:j+2] * ft_mask_2)

    u = np.zeros(fx.shape)
    v = np.zeros(fx.shape)

    u_new = np.zeros(fx.shape)
    v_new = np.zeros(fx.shape)

    opt_flow = np.zeros(fx.shape)

    for iter in range(0,10):
        print("starting iteration: " + str(iter))
        for i in range(1,fx.shape[0]-1):
            for j in range(1,fx.shape[1]-1):
                u_av = (np.sum(u[i-1:i+2,j-1:j+2]) - u[i,j])/8
                v_av = (np.sum(v[i-1:i+2,j-1:j+2]) - v[i,j])/8
                P = fx[i,j]*u_av + fy[i,j]*v_av + ft[i,j]
                D = lamda + fx[i,j]**2 + fy[i,j]**2
                u_new[i,j] = u_av - ((fx[i,j]*P)/D)
                v_new[i,j] = v_av - ((fy[i,j]*P)/D)
        u = u_new
        v = v_new
        print("error: " + str(np.average(fx*u+fy*v+ft)))
    opt_flow = np.sqrt(u**2 + v**2)

    hsv = np.zeros((img_1.shape[0],img_1.shape[1],3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    scale = img_1_output.shape[0]/img_1.shape[0]
    for i in range(0,img_1_output.shape[0]-scale,4):
        for j in range(2,img_1_output.shape[1]-scale,4):
            if opt_flow[i/scale,j/scale] > 1:
                cv2.arrowedLine(img_1_output, (j,i), (j+4*int(u[i/scale,j/scale]),i+4*int(v[i/scale,j/scale])), color=tuple(bgr[i/scale,j/scale,:].tolist()), thickness=1, tipLength=0.2)

    cv2.imwrite(file_name+str(file_str_no)+'opti.jpg',opt_flow)
    cv2.imwrite(file_name+str(file_str_no)+'opti_color.jpg',bgr)
    cv2.imwrite(file_name+str(file_str_no)+'opti_arrow.jpg',img_1_output)

#     cv2.namedWindow('img_1', cv2.WINDOW_NORMAL)
#     cv2.namedWindow('opt_flow', cv2.WINDOW_NORMAL)
#     cv2.namedWindow('opti_arrow', cv2.WINDOW_NORMAL)

#     cv2.imshow('img_1', img_1)
#     cv2.imshow('opt_flow', bgr)
#     cv2.imshow('opti_arrow', img_1_output)
# # plt.imshow(opt_flow)
# # plt.show()
#     cv2.waitKey(0)

cv2.destroyAllWindows()
