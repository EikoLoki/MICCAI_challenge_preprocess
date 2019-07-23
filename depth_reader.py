import cv2
from os import listdir
from os.path import join, isfile, normpath
import numpy as np
import matplotlib.pyplot as plt
import json
import tifffile as tiff


def read_camera_para(parameter_file):
    with open(parameter_file) as para_json_file:
        data = json.load(para_json_file)
        camera_para = data['camera-calibration']

        l_camera_matrix = np.array(camera_para['KL'])
        r_camera_matrix = np.array(camera_para['KR'])
        l_dist_coeff = np.array(camera_para['DL'])
        r_dist_coeff = np.array(camera_para['DR'])

    return l_camera_matrix, r_camera_matrix, l_dist_coeff, r_dist_coeff



def tiff_reader(tiff_file):
    raw = tiff.imread(tiff_file)
    img_l = raw[:1024,:,:]
    img_r = raw[1024:,:,:]
    print(img_l.shape, img_l.dtype)
    print(img_r.shape, img_r.dtype)

    z_l = img_l[:,:,2]
    z_r = img_r[:,:,2]

    z_l[z_l > 500] = 0
    z_r[z_r > 500] = 0
    z_l[z_l < 20] = 0
    z_r[z_r < 20] = 0

    z_l = (z_l - z_l.min()) / z_l.max() * 255
    z_r = (z_r - z_r.min()) / z_r.max() * 255
    #plt.imshow(z)
    #plt.show()

    """
    base_line = 4.14339
    focal_length = 1035

    disp = np.zeros(z.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i, j] < 200 and z[i,j] > 50 :
                disp[i, j] = base_line * focal_length / z[i,j]

    print(disp.shape)

    #plt.imshow(disp)
    #plt.show()
    """


    return img_l, img_r

def reprojection(PointCloud, l_camera_matrix, l_dist_coeff):
    rvec = np.zeros((3,1))
    tvec = np.zeros((3,1))
    size = PointCloud.shape[:2]
    PointCloud = PointCloud.reshape(-1, 3)

    img, jacobian = cv2.projectPoints(PointCloud, rvec, tvec, l_camera_matrix, l_dist_coeff)
    img = np.squeeze(img)

    reproject_img = np.zeros(size)
    print(reproject_img.shape)
    for i in range(img.shape[0]):
        x = int(img[i,0])
        y = int(img[i,1])
        if x < size[1] and x > 0 and y < size[0] and y > 0:
            reproject_img[y,x] = PointCloud[i,2]


    count = 0
    for i in range(reproject_img.shape[0]):
        for j in range(reproject_img.shape[1]):
            if reproject_img[i,j] == 0:
                count += 1

    print('hollow point:', count)
    plt.imshow(reproject_img)
    plt.show()
    return reproject_img


def main():

    rootpath = '/media/xiran_zhang/TOSHIBA EXT/MICCAI_SCARED/dataset3'
    testpath = 'keyframe_1/data/scene_points/scene_points000000.tiff'
    frame_data_file = 'keyframe_1/data/frame_data/frame_data000000.json'
    depth_file = join(rootpath, testpath)
    camera_file = join(rootpath, frame_data_file)

    l_camera_matrix, r_camera_matrix, l_dist_coeff, r_dist_coeff = read_camera_para(camera_file)
    img_l, img_r = tiff_reader(depth_file)
    repro_l = reprojection(img_l, l_camera_matrix, l_dist_coeff)
    repro_r = reprojection(img_r, r_camera_matrix, r_dist_coeff)


    save_path = 'keyframe_2/data/test_data000200.png'

    cv2.imshow('test', repro_l)
    cv2.waitKey()

    #cv2.imwrite(join(rootpath,save_path),repro_l)

    """
    l_camera_matrix, r_camera_matrix, l_dist_coeff, r_dist_coeff = read_camera_para(camera_file)
    build_depth_image(depth_file, l_camera_matrix,r_camera_matrix,l_dist_coeff,r_dist_coeff)
    """


if __name__ == '__main__':
    main()