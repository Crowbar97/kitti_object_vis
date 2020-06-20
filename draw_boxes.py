from os import path, listdir
import argparse

import numpy as np
import cv2
from tqdm import tqdm

import kitti_util as utils


def get_image(idx):
    img_filename = path.join(image_dir, "%06d.png" % (idx))
    return utils.load_image(img_filename)


def get_label_objects(idx):
    label_filename = path.join(label_dir, "%06d.txt" % (idx))
    return utils.read_label(label_filename)


def get_calibration(idx):
    calib_filename = path.join(calib_dir, "%06d.txt" % (idx))
    return utils.Calibration(calib_filename)


def norm_score(score):
    if abs(score) > 10:
        score = 10 * np.sign(score)
    return (score + 10) / 20 * 100


def show_image_with_boxes(data_idx, img,
                          objects, calib,
                          show3d=True, depth=None):
    img_res = np.copy(img)
    for obj in objects:
        if obj.type == "DontCare":
            continue

        obj.score = norm_score(obj.score)
        if obj.score < args.threshold:
            continue

        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img_res = utils.draw_projected_box3d(img_res, box3d_pts_2d, obj)

    cv2.imwrite(path.join(output_dir, '%.3d_image_with_boxes.png' % data_idx), img_res)


def draw(data_idx):
    img = get_image(data_idx)
    objects = get_label_objects(data_idx)
    calib = get_calibration(data_idx)

    show_image_with_boxes(data_idx, img, objects, calib)


parser = argparse.ArgumentParser(description='PCDet framework result visualizer')
# parser.add_argument('label_dir_path', type=str,
#                     help='Path to directory with input labels')
# parser.add_argument('output_dir_path', type=str,
#                     help='Path to output dir for result image storage')
parser.add_argument('-t', '--threshold', type=float, default=0,
                    help='Confidence threshold')
args = parser.parse_args()


# image_dir = 'data/object/training/image_2'
image_dir = '/home/crowbar/PCDet/data/kitti/training/image_2'

# calib_dir = 'data/object/training/calib'
calib_dir = '/home/crowbar/PCDet/data/kitti/training/calib'

# label_dir = 'data/object/training/label_2'
# label_dir = args.label_dir_path
label_dir = '/home/crowbar/pc_det_aux/result_parsing/ready/second_mots/0001'

# output_dir = args.output_dir_path
output_dir = 'result/second_mots/0001'


def main():
    for file_name in tqdm(listdir(path.join(label_dir))[:]):
        idx = int(file_name[:-4])
        # tqdm.write(str(idx))
        draw(idx)


main()
