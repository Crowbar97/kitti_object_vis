from os import path, listdir
import argparse

import mayavi.mlab as mlab
mlab.options.offscreen = True

import numpy as np
import cv2
from tqdm import tqdm

import kitti_util as utils
from viz_util import draw_lidar_simple, draw_lidar, draw_gt_box3d


def get_label_objects(idx):
    label_filename = path.join(label_dir, "%06d.txt" % (idx))
    return utils.read_label(label_filename)


def get_calibration(idx):
    calib_filename = path.join(calib_dir, "%06d.txt" % (idx))
    return utils.Calibration(calib_filename)


def get_lidar(idx, dtype=np.float64, n_vec=4):
    lidar_filename = path.join(lidar_dir, "%06d.bin" % (idx))
    return utils.load_velo_scan(lidar_filename, dtype, n_vec)


def norm_score(score):
    if abs(score) > 10:
        score = 10 * np.sign(score)
    return (score + 10) / 20 * 100


def show_lidar_with_depth(data_idx,
                          pc_velo,
                          objects,
                          calib,
                          fig,
                          objects_pred=None,
                          depth=None,
                          constraint_box=False,
                          pc_label=False,
                          save=False):
    print(("All point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue

        obj.score = norm_score(obj.score)
        if obj.score < args.threshold:
            continue

        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_box3d(box3d_pts_3d_velo, fig, obj)

    mlab.view(azimuth=180,
              elevation=65,
              # focalpoint=[12.0909996 , -1.04700089, -2.03249991],
              focalpoint=[0, 0, 0],
              distance=40.0,
              figure=fig
              )

    # mlab.show(stop=False)

    mlab.savefig(filename=path.join(output_dir, 'pc_%.3d.png' % data_idx),
                 figure=fig
                 )


def show(idx):
    objects = get_label_objects(idx)
    calib = get_calibration(idx)

    fig = mlab.figure(figure=None,
                      bgcolor=(0, 0, 0),
                      fgcolor=None,
                      engine=None,
                      size=(1242, 375))

    dtype = np.float32
    n_vec = 4
    pc_velo = get_lidar(idx, dtype, n_vec)[:, 0:n_vec]

    objects_pred = None
    depth = None
    const_box = False
    save_depth = False
    pc_label = False

    show_lidar_with_depth(idx,
                          pc_velo,
                          objects,
                          calib,
                          fig,
                          objects_pred,
                          depth,
                          constraint_box=const_box,
                          save=save_depth,
                          pc_label=pc_label)



parser = argparse.ArgumentParser(description='PCDet framework result point cloud visualizer')
# parser.add_argument('label_dir_path', type=str,
#                     help='Path to directory with input labels')
# parser.add_argument('output_dir_path', type=str,
#                     help='Path to output dir for result pc scan storage')
parser.add_argument('-t', '--threshold', type=float, default=0,
                    help='Confidence threshold')
args = parser.parse_args()


# lidar_dir = 'data/object/training/velodyne'
lidar_dir = '/home/crowbar/PCDet/data/kitti/training/velodyne'

# calib_dir = 'data/object/training/calib'
calib_dir = '/home/crowbar/PCDet/data/kitti/training/calib'

# label_dir = 'data/object/training/label_2'
# label_dir = args.label_dir_path
label_dir = '/home/crowbar/pc_det_aux/result_parsing/ready/second_mots/0001'

# output_dir = args.output_dir_path
output_dir = 'result/second_mots_pc/0001'


def main():
    for file_name in tqdm(listdir(path.join(label_dir))[:]):
        idx = int(file_name[:-4])
        # tqdm.write(str(idx))
        show(idx)

main()
