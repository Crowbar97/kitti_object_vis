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
    label_filename = path.join(label_dir, args.input_folder_name, "%06d.txt" % (idx))
    return utils.read_label(label_filename)


def get_calibration(idx):
    calib_filename = path.join(calib_dir, "%06d.txt" % (idx))
    return utils.Calibration(calib_filename)

# TODO: add labels
def new_show():
    if objects_pred is not None:
        color = (255, 0, 0)
        for obj in objects_pred:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                1,
            )
        startx = 165
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [obj.type for obj in objects_pred if obj.type in type_list]
        text_lables.insert(0, "3D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(
                img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
            )

    cv2.imshow("with_bbox", img1)
    cv2.imwrite("imgs/" + str(name) + ".png", img1)


# OpenCV uses BGR format
class_colors = {
    'Pedestrian': (0, 0, 255),
    'Cyclist': (255, 0, 0),
    'Car': (0, 255, 0)
    }


def show_image_with_boxes(data_idx, img,
                          objects, calib,
                          show3d=True, depth=None):
    img_res = np.copy(img)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img_res = utils.draw_projected_box3d(img_res, box3d_pts_2d,
                                             color=class_colors[obj.type])

    cv2.imwrite(path.join(args.output_dir_path, '%s_image_with_boxes.png' % data_idx), img_res)


def draw(data_idx):
    img = get_image(data_idx)
    objects = get_label_objects(data_idx)
    calib = get_calibration(data_idx)

    show_image_with_boxes(data_idx, img, objects, calib)


image_dir = 'data/object/training/image_2'
label_dir = 'data/object/training/label_2'
calib_dir = 'data/object/training/calib'


parser = argparse.ArgumentParser(description='PCDet framework result visualizer')
parser.add_argument('input_folder_name', type=str,
                    help='Folder name with input labels')
parser.add_argument('output_dir_path', type=str,
                    help='Path to output dir for result images storage')
args = parser.parse_args()


def main():
    for file_name in tqdm(listdir(path.join(label_dir, args.input_folder_name))[:]):
        idx = int(file_name[:-4])
        # tqdm.write(str(idx))
        draw(idx)


main()
