import cv2
from os import listdir
from os.path import join, split
import numpy as np


def image_scissor(image_file, left_savepath, right_savepath):
    print('-- current image :' + image_file + " --")
    stacked = cv2.imread(image_file)
    print(stacked.shape)
    left_img = stacked[:1024, :, :]
    right_img = stacked[1024:, :, :]
    path, file = split(image_file)

    cv2.imwrite(join(left_savepath, file), left_img)
    cv2.imwrite(join(right_savepath, file), right_img)




def main():
    rootpath = '/home/eikoloki/Documents/MICCAI_SCARED/dataset1'
    keyframe_list = [join(rootpath, kf) for kf in listdir(rootpath) if ('keyframe' in kf and 'ignore' not in kf)]
    for kf in keyframe_list:
        stacked_filepath = join(rootpath, kf) + '/data/rgb_data'
        stacked_filelist = [sf for sf in listdir(stacked_filepath) if '.png' in sf]
        for sf in stacked_filelist:
            image_file = join(stacked_filepath, sf)
            left_savepath = join(rootpath, kf) + '/data/left'
            right_savepath = join(rootpath, kf) + '/data/right'
            image_scissor(image_file, left_savepath, right_savepath)


if __name__ == '__main__':
    main()

