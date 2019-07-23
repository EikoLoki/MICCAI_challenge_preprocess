import cv2
import json
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

def tiff_reader(tiff_file):
    raw = tiff.imread(tiff_file)
    scene_l = raw[:1024,:,:]
    img_r = raw[1024:,:,:]
    print("left point cloud shape:", scene_l.shape, scene_l.dtype)

    return scene_l # return value is a cv mat



def camera_pose_reader(parameter_file):
    with open(parameter_file) as para_json_file:
        data = json.load(para_json_file)
        camera_pose = data['camera-pose']

        pose_transformation = np.array(camera_pose)

    return pose_transformation # return value is an np array


def camera_param_reader(parameter_file):
    with open(parameter_file) as para_json_file:
        data = json.load(para_json_file)
        camera_para = data['camera-calibration']

        l_camera_matrix = np.array(camera_para['KL'])
        #r_camera_matrix = np.array(camera_para['KR'])
        l_dist_coeff = np.array(camera_para['DL'])
        #r_dist_coeff = np.array(camera_para['DR'])

    return l_camera_matrix, l_dist_coeff #, r_camera_matrix, r_dist_coeff


def view_transfer(camera_pose_1, camera_pose_2, point_cloud_1, image_1):
    camera_pose_2_inv = np.linalg.inv(camera_pose_2)
    transformation = np.transpose(np.dot(camera_pose_2_inv, camera_pose_1))
    print(image_1.shape)
    h,w = image_1.shape[:2]
    homo_map = np.ones((h,w,1))
    homo_point_cloud_1= np.concatenate((point_cloud_1, homo_map), axis=2)
    transformed_point_cloud = np.tensordot(homo_point_cloud_1, transformation, axes=([2], [1]))
    transformed_point_cloud = np.array(transformed_point_cloud[:,:,:3])

    return transformed_point_cloud, image_1

def reprojection_img(transformed_point_cloud, image_original, l_camera_matrix, l_dist_coeff):
    rvec = np.zeros((3,1))
    tvec = np.zeros((3,1))
    h,w = image_original.shape[:2]
    print('transformed pc shape:', transformed_point_cloud.shape, transformed_point_cloud.dtype)
    PC = transformed_point_cloud.reshape(-1, 3)
    image_original_vector = image_original.reshape(-1,3)
    repro_coor, jacobian = cv2.projectPoints(PC, rvec, tvec, l_camera_matrix, l_dist_coeff)
    repro_coor = np.squeeze(repro_coor)

    reproject_img = np.zeros([h,w,3])
    for i in range(repro_coor.shape[0]):
        x = int(repro_coor[i,0])
        y = int(repro_coor[i,1])
        if x < w and x > 0 and y < h and y > 0:
            reproject_img[y,x,0] = image_original_vector[i,0]
            reproject_img[y,x,1] = image_original_vector[i,1]
            reproject_img[y,x,2] = image_original_vector[i,2]


    count = 0
    for i in range(reproject_img.shape[0]):
        for j in range(reproject_img.shape[1]):
            if reproject_img[i,j,2] == 0:
                count += 1

    print('hollow point:', count)

    return reproject_img


def reprojection_depth(transformed_point_cloud, l_camera_matrix, l_dist_coeff):
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    h, w = transformed_point_cloud.shape[:2]
    print('transformed pc shape:', transformed_point_cloud.shape, transformed_point_cloud.dtype)
    PC = transformed_point_cloud.reshape(-1, 3)
    repro_coor, jacobian = cv2.projectPoints(PC, rvec, tvec, l_camera_matrix, l_dist_coeff)
    repro_coor = np.squeeze(repro_coor)

    reproject_depth = np.zeros([h, w, 3])
    for i in range(repro_coor.shape[0]):
        x = int(repro_coor[i, 0])
        y = int(repro_coor[i, 1])
        if x < w and x > 0 and y < h and y > 0:
            reproject_depth[y, x, 0] = PC[i, 0]
            reproject_depth[y, x, 1] = PC[i, 1]
            reproject_depth[y, x, 2] = PC[i, 2]

    count = 0
    for i in range(reproject_depth.shape[0]):
        for j in range(reproject_depth.shape[1]):
            if reproject_depth[i, j, 2] == 0:
                count += 1

    return  reproject_depth


def get_scene_pair(pointcloud_filepath, camera_filepath, current_num,forward_step):
    current_scene = tiff_reader(join(pointcloud_filepath, "scene_points%.6d.tiff" % current_num))
    next_scene = tiff_reader(join(pointcloud_filepath, "scene_points%.6d.tiff" % (current_num + forward_step)))

    current_camera_matrix, current_dist_coeff = camera_param_reader(join(camera_filepath, "frame_data%.6d.json" % current_num))
    next_camera_matrix, next_dist_coeff = camera_param_reader(join(camera_filepath, "frame_data%.6d.json" % (current_num + forward_step)))

    current_pose = camera_pose_reader(join(camera_filepath, "frame_data%.6d.json" % current_num))
    next_pose = camera_pose_reader(join(camera_filepath, "frame_data%.6d.json" % (current_num + forward_step)))

    return current_scene, next_scene, current_pose, next_pose, current_camera_matrix, current_dist_coeff, next_camera_matrix, next_dist_coeff

def get_img_pair(image_filepath, current_num, forward_step):
    current_img = cv2.imread(join(image_filepath,'frame_data%.6d.png' % current_num))
    next_img = cv2.imread(join(image_filepath, 'frame_data%.6d.png' % (current_num + forward_step)))

    return current_img, next_img

def main():


    rootpath = '/media/xiran_zhang/TOSHIBA EXT/MICCAI_SCARED/dataset3'
    camera_filepath = join(rootpath, 'keyframe_1/data/frame_data')
    pointcloud_filepath = join(rootpath, 'keyframe_1/data/scene_points')
    img_filepath = join(rootpath, 'keyframe_1/data/left_finalpass')



    current_num  = 0
    forward_step = 250
    pc1, pc2, pose1, pose2, camera_para1, dist_coeff1, camera_para2, dist_coeff2 = get_scene_pair(pointcloud_filepath, camera_filepath, current_num, forward_step)
    img1, img2 = get_img_pair(img_filepath, current_num, forward_step)


    transformed_pc1, img_original = view_transfer(pose1, pose2, pc1, img1)

    reproject_pointcloud = reprojection_depth(transformed_pc1, camera_para2, dist_coeff2)
    tiff.imsave('current_transformed_pc.tiff', reproject_pointcloud.astype(np.float32))
    cv2.imwrite('current_transformed_frame.png', img_original)
    tiff.imsave('current_raw_pc.tiff', pc2.astype(np.float32))
    cv2.imwrite('current_raw_frame.png',img2)

    reproject_img2 = reprojection_img(reproject_pointcloud, img_original, camera_para2, dist_coeff2)
    img2 = reprojection_img(pc2, img2, camera_para2, dist_coeff2)
    cv2.imwrite('repro_img_2.png', reproject_img2)
    cv2.imwrite('img_2.png', img2)
    cv2.imwrite('diff.png', reproject_img2 / 2 + img2 / 2)
    """
    transformed_cloud, img_original = view_transfer(pose1, pose2, pc1, img1)

    reprojected_to_img2 = reprojection(transformed_cloud, img1, camera_para2, dist_coeff2)

    cv2.imwrite('reprojected_to_2.png', reprojected_to_img2)

    original_img1 = reprojection(pc1, img1, camera_para1, dist_coeff1)

    cv2.imwrite('current_image.png', original_img1)

    # reproject the original one
    original_img2 = reprojection(pc2, img2, camera_para2, dist_coeff2)

    cv2.imwrite('next_image.png', original_img2)

    cv2.imwrite('difference.png', original_img2/2 + reprojected_to_img2/2)
    """


if __name__ == '__main__':
    main()
