from video_parser import video_parser
from image_scissor import image_scissor
from stereo_rectify import stereo_rectify
from depth_to_disp import depth_to_disparity

rootpath = '/media/eikoloki/TOSHIBA EXT/MICCAI_SCARED/dataset3'

video_parser(rootpath)

image_scissor(rootpath)

stereo_rectify(rootpath)

depth_to_disparity(rootpath)
