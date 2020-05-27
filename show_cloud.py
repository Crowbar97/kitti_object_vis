from os import path, listdir
import argparse

import mayavi.mlab as mlab
import numpy as np
import cv2
from tqdm import tqdm

import kitti_util as utils
from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d


def get_image(idx):
    img_filename = path.join(image_dir, "%06d.png" % (idx))
    return utils.load_image(img_filename)


def get_label_objects(idx):
    # FIXME
    label_filename = path.join(label_dir, 'part', "%06d.txt" % (idx))
    return utils.read_label(label_filename)


def get_calibration(idx):
    calib_filename = path.join(calib_dir, "%06d.txt" % (idx))
    return utils.Calibration(calib_filename)


def get_lidar(idx, dtype=np.float64, n_vec=4):
    lidar_filename = path.join(lidar_dir, "%06d.bin" % (idx))
    return utils.load_velo_scan(lidar_filename, dtype, n_vec)


def show_lidar_with_depth(data_idx,
                          pc_velo,
                          objects,
                          calib,
                          fig,
                          img_fov=False,
                          img_width=None,
                          img_height=None,
                          objects_pred=None,
                          depth=None,
                          cam_img=None,
                          constraint_box=False,
                          pc_label=False,
                          save=False):
    print(("All point num: ", pc_velo.shape[0]))
    # if img_fov:
    #     pc_velo_index = get_lidar_index_in_image_fov(
    #         pc_velo[:, :3], calib, 0, 0, img_width, img_height
    #     )
    #     pc_velo = pc_velo[pc_velo_index, :]
    #     print(("FOV point num: ", pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    # Draw depth
    # if depth is not None:
    #     depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

    #     indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
    #     depth_pc_velo = np.hstack((depth_pc_velo, indensity))
    #     print("depth_pc_velo:", depth_pc_velo.shape)
    #     print("depth_pc_velo:", type(depth_pc_velo))
    #     print(depth_pc_velo[:5])
    #     draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

    #     if save:
    #         data_idx = 0
    #         vely_dir = "data/object/training/depth_pc"
    #         save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))
    #         print(save_filename)
    #         # np.save(save_filename+".npy", np.array(depth_pc_velo))
    #         depth_pc_velo = depth_pc_velo.astype(np.float32)
    #         depth_pc_velo.tofile(save_filename)

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, label=obj.type)

    # if objects_pred is not None:
    #     color = (1, 0, 0)
    #     for obj in objects_pred:
    #         if obj.type == "DontCare":
    #             continue
    #         # Draw 3d bounding box
    #         box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
    #         box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    #         print("box3d_pts_3d_velo:")
    #         print(box3d_pts_3d_velo)
    #         draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
    #         # Draw heading arrow
    #         ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
    #         ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
    #         x1, y1, z1 = ori3d_pts_3d_velo[0, :]
    #         x2, y2, z2 = ori3d_pts_3d_velo[1, :]
    #         mlab.plot3d(
    #             [x1, x2],
    #             [y1, y2],
    #             [z1, z2],
    #             color=color,
    #             tube_radius=None,
    #             line_width=1,
    #             figure=fig,
    #         )
    mlab.show(stop=False)
    # mlab.savefig(filename='lidar' + str(data_idx) + '.png')


image_dir = 'data/object/training/image_2'
label_dir = 'data/object/training/label_2'
calib_dir = 'data/object/training/calib'
lidar_dir = 'data/object/training/velodyne'


parser = argparse.ArgumentParser(description='PCDet framework result point cloud visualizer')
parser.add_argument('scene_id', type=int,
                    help='Scene id')
args = parser.parse_args()


def main():
    img = get_image(args.scene_id)
    img_height, img_width, img_channel = img.shape

    objects = get_label_objects(args.scene_id)
    calib = get_calibration(args.scene_id)

    fig = mlab.figure(figure=None,
                      bgcolor=(0, 0, 0),
                      fgcolor=None,
                      engine=None,
                      size=(1000, 500))

    dtype = np.float32
    n_vec = 4
    pc_velo = get_lidar(args.scene_id, dtype, n_vec)[:, 0:n_vec]

    img_fov = False
    objects_pred = None
    depth = None
    const_box = False
    save_depth = False
    pc_label = False

    show_lidar_with_depth(args.scene_id,
                          pc_velo,
                          objects,
                          calib,
                          fig,
                          img_fov,
                          img_width,
                          img_height,
                          objects_pred,
                          depth,
                          img,
                          constraint_box=const_box,
                          save=save_depth,
                          pc_label=pc_label)


main()
