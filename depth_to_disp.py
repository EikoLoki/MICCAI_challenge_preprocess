import cv2
import json
from os.path import join, splitext, split
from os import listdir
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

def tiff_reader(tiff_file):
    raw = tiff.imread(tiff_file)
    img_l = raw[:1024,:,:]
    img_r = raw[1024:,:,:]

    return img_l, img_r

def coor_to_disp(coor, Q):

    # parse Q
    Q = np.array(Q)
    fl = Q[2,3]
    bl =  1 / Q[3,2]
    cx = -Q[0,3]
    cy = -Q[1,3]

    print('fl: ', fl, 'bl: ', bl, 'cx: ', cx, 'cy: ', cy)

    size = coor.shape[:2] # size[0] = 1024 size[1] = 1280
    X = coor[:,:,0]
    Y = coor[:,:,1]
    Z = coor[:,:,2]

    #disp = np.zeros(size)

    all_disp = np.zeros((size[0],size[1],2))
    disp = np.zeros(size)

    for i in range(size[0]):
        for j in range(size[1]):
            x = X[i,j]
            y = Y[i,j]
            z = Z[i,j]

            if (z != 0):
                d = fl * bl / z
                p_x = fl * x / z + cx
                p_y = fl * y / z + cy

                if (p_x < size[1] and p_y < size[0]):
                    all_disp[int(p_y), int(p_x),0] += d
                    all_disp[int(p_y), int(p_x),1] += 1

                """
                if (p_x <= size[1] and p_y <= size[0]):
                    print('px: ', p_x, 'py: ', p_y, 'disp: ', d)
                    disp[int(p_y), int(p_x)] = d
                """
    for i in range(size[0]):
        for j in range(size[1]):
            if all_disp[i,j,1] != 0:
                disp[i,j] = all_disp[i,j,0] / all_disp[i,j,1]




    #print(disp.max(), disp.min())
    #plt.imshow(disp)
    #plt.show()
    return disp

def read_Q(reprojection_file):

    with open(reprojection_file) as json_file:
        data = json.load(json_file)
        Q = data['reprojection-matrix']

        return  Q


def main(path):
    rootpath = path
    keyframe_list = ['keyframe_4']
    for kf in keyframe_list:
        reprojection_filepath = join(rootpath, kf) + '/data/reprojection_data'
        coor_filepath = join(rootpath,kf) + '/data/scene_points'
        disp_filepath = join(rootpath,kf) + '/data/disparity'
        frame_list = listdir(reprojection_filepath)

        for i in range(len(frame_list)):
            reprojection_data = reprojection_filepath + '/frame_data%.6d.json' % i
            coor_data = coor_filepath + '/scene_points%.6d.tiff' % i
            disp_data = disp_filepath + '/frame_data%.6d.tiff' % i
            print('Saving disparity to:', disp_data)

            Q = read_Q(reprojection_data)
            img_l, img_r = tiff_reader(coor_data)
            disp = coor_to_disp(img_l, Q)
            cv2.imwrite(disp_data, disp)




if __name__ == '__main__':
    path = '/media/eikoloki/TOSHIBA EXT/MICCAI_SCARED/dataset2'
    main(path)