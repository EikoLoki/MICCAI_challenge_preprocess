import cv2
from os.path import join
import tifffile as tiff
import matplotlib.pyplot as plt
import json
import numpy as np


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


def reprojection(PointCloud_pool, l_camera_matrix, l_dist_coeff):
    rvec = np.zeros((3,1))
    tvec = np.zeros((3,1))
    h,w = PointCloud_pool.shape[1:3]
    PointCloud = PointCloud_pool.reshape(-1, 3)
    print(PointCloud.shape)
    img, jacobian = cv2.projectPoints(PointCloud, rvec, tvec, l_camera_matrix, l_dist_coeff)
    img = np.squeeze(img)

    reproject_img = np.zeros([h,w,3])
    for i in range(img.shape[0]):
        x = int(img[i,0])
        y = int(img[i,1])
        if x < w and x > 0 and y < h and y > 0:
            reproject_img[y,x,0] = PointCloud[i,0]
            reproject_img[y,x,1] = PointCloud[i,1]
            reproject_img[y,x,2] = PointCloud[i,2]


    count = 0
    for i in range(reproject_img.shape[0]):
        for j in range(reproject_img.shape[1]):
            if reproject_img[i,j,2] == 0:
                count += 1

    print('hollow point:', count)
    print(reproject_img.shape)
    plt.imshow(reproject_img[:,:,2])
    plt.show()
    return reproject_img



# suggest forward_frame_num 20, backward_frame_num 5
def fusion_frame(camera_filepath, scene_filepath, current_frame, forward_frame_num, backward_frame_num):
    # stack all the data in the list (scene, camera pose)
    current_scene = tiff_reader(join(scene_filepath, "scene_points%.6d.tiff" % current_frame))
    print('current scene shape:', current_scene.shape)
    current_pose = camera_pose_reader(join(camera_filepath, "frame_data%.6d.json" % current_frame))
    forward_scene_pool = []
    backward_scene_pool = []
    forward_pose_pool = []
    backward_pose_pool = []

    for n in range(forward_frame_num):
        forward_scene = tiff_reader(join(scene_filepath, "scene_points%.6d.tiff" % (current_frame + n)))
        forward_pose = camera_pose_reader(join(camera_filepath, "frame_data%.6d.json" % (current_frame + n)))
        forward_scene_pool.append(forward_scene)
        forward_pose_pool.append(forward_pose)

    for m in range(backward_frame_num):
        backward_scene = tiff_reader(join(scene_filepath, "scene_points%.6d.tiff" % (current_frame - m)))
        backward_pose = camera_pose_reader(join(camera_filepath, "frame_data%.6d.json" % (current_frame - m)))
        backward_scene_pool.append(backward_scene)
        backward_pose_pool.append(backward_pose)


    current_g_inv = np.linalg.inv(current_pose)
    PointCloud_pool = []
    # operating forward first
    for n in range(len(forward_pose_pool)):
        forward_g = forward_pose_pool[n]
        forward_transformation = np.transpose(np.dot(current_g_inv, forward_g))

        scene = forward_scene_pool[n]
        h,w = scene.shape[:2]
        homo_map = np.ones((h,w,1))
        scene = np.concatenate((scene,homo_map), axis=2)
        forward_transformed_scene = np.tensordot(scene, forward_transformation, axes=([2],[1]))

        print("forward transformed scene shape:", forward_transformed_scene.shape, forward_transformed_scene.dtype)
        PointCloud_pool.append(forward_transformed_scene[:,:,:3])

    # operating backward sequence
    backward_transformed_scene_pool = []
    for m in range(len(backward_pose_pool)):
        backward_g = backward_pose_pool[m]
        backward_transformation = np.transpose(np.dot(current_g_inv, backward_g))

        scene = backward_scene_pool[m]
        h, w = scene.shape[:2]
        homo_map = np.ones((h, w, 1))
        scene = np.concatenate((scene, homo_map), axis=2)
        backward_transformed_scene = np.tensordot(scene, backward_transformation, axes=([2], [1]))

        print("backward transformed scene shape:", backward_transformed_scene.shape, backward_transformed_scene.dtype)
        PointCloud_pool.append(backward_transformed_scene[:, :, :3])

    PointCloud_pool.append(current_scene)
    PointCloud_pool = np.array(PointCloud_pool)
    print(PointCloud_pool.shape)

    return PointCloud_pool


def main():
    rootpath = '/media/xiran_zhang/TOSHIBA EXT/MICCAI_SCARED/dataset3'
    camera_filepath = join(rootpath, 'keyframe_1/data/frame_data')
    scene_filepath = join(rootpath, 'keyframe_1/data/scene_points')

    PointCloud_pool = fusion_frame(camera_filepath, scene_filepath, 6, 10, 5)
    l_camera_matrix, l_dist_coeff = camera_param_reader(join(camera_filepath, 'frame_data%.6d.json' % 6))
    reproject_img = reprojection(PointCloud_pool, l_camera_matrix, l_dist_coeff)
    tiff.imsave(join(rootpath, 'keyframe_1/reproj_img.tiff'), reproject_img.astype(np.float32))

if __name__ == '__main__':
    main()