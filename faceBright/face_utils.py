# coding: utf-8

import cv2
import os
import numpy as np
import pickle
import math
from skimage.feature import hog

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def mkdir(d):
    os.makedirs(d, exist_ok=True)

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def draw_cali_circle(window_img, center_pt, size_factor, line_size, timer, start_time):
    cv2.circle(window_img, center_pt, int(size_factor) +1, (255,0,0), 5, cv2.LINE_AA)
    cv2.circle(window_img, center_pt, int((size_factor/3)*(timer-start_time)), (0,0,255), -1, cv2.LINE_AA)
    cv2.line(window_img, (center_pt[0] - line_size, center_pt[1]), (center_pt[0] + line_size, center_pt[1]), (0,255,0), 2, cv2.LINE_AA)
    cv2.line(window_img, (center_pt[0], center_pt[1] - line_size), (center_pt[0], center_pt[1] + line_size), (0,255,0), 2, cv2.LINE_AA)

def recon_vers(param_lst, roi_box_lst, u_base, w_shp_base, w_exp_base):
    ver_lst = []
    for param, roi_box in zip(param_lst, roi_box_lst):
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        pts3d = R @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp). \
            reshape(3, -1, order='F') + offset
        pts3d = similar_transform(pts3d, roi_box, 120)

        ver_lst.append(pts3d)

    return ver_lst

def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)

def _parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    trans_dim, shape_dim, exp_dim = 12, 40, 10

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr

# To calculate head gaze by 3 rigid facial point
def surface_normal_cross(poly):
    n = np.cross(poly[1,:]-poly[0,:],poly[2,:]-poly[0,:])
    norm = np.linalg.norm(n)
    if norm==0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm
    return n , normalised

def dist_2D(p1, p2):
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist

def calc_unit_vec(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError('zero norm')
    else:
        normalised = vec/norm
    return normalised


def align_face_68pts(img, img_land, box_enlarge, img_size=112):
    """Performs affine transformation to align the images by eyes.

    Performs affine alignment including eyes.

    Args:
        img: gray or RGB
        img_land: 68 system flattened landmarks, shape:(136)
        box_enlarge: relative size of face on the image. Smaller value indicate larger proportion
        img_size = output image size
    Return:
        aligned_img: the aligned image
        new_land: the new landmarks
    """
    leftEye0 = (img_land[2 * 36] + img_land[2 * 37] + img_land[2 * 38] + img_land[2 * 39] + img_land[2 * 40] +
                img_land[2 * 41]) / 6.0
    leftEye1 = (img_land[2 * 36 + 1] + img_land[2 * 37 + 1] + img_land[2 * 38 + 1] + img_land[2 * 39 + 1] +
                img_land[2 * 40 + 1] + img_land[2 * 41 + 1]) / 6.0
    rightEye0 = (img_land[2 * 42] + img_land[2 * 43] + img_land[2 * 44] + img_land[2 * 45] + img_land[2 * 46] +
                 img_land[2 * 47]) / 6.0
    rightEye1 = (img_land[2 * 42 + 1] + img_land[2 * 43 + 1] + img_land[2 * 44 + 1] + img_land[2 * 45 + 1] +
                 img_land[2 * 46 + 1] + img_land[2 * 47 + 1]) / 6.0
    deltaX = (rightEye0 - leftEye0)
    deltaY = (rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
    mat2 = np.mat([[leftEye0, leftEye1, 1], [rightEye0, rightEye1, 1], [img_land[2 * 30], img_land[2 * 30 + 1], 1],
                   [img_land[2 * 48], img_land[2 * 48 + 1], 1], [img_land[2 * 54], img_land[2 * 54 + 1], 1]])
    mat2 = (mat1 * mat2.T).T
    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1
    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    land_3d = np.ones((int(len(img_land) / 2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land) / 2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.array(list(zip(new_land[:, 0], new_land[:, 1]))).astype(int)

    return aligned_img, new_land


def extract_hog(frame, orientation=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Extract HOG features from a frame.

    Args:
        frame (array]): Frame of image]
        orientation (int, optional): Orientation for HOG. Defaults to 8.
        pixels_per_cell (tuple, optional): Pixels per cell for HOG. Defaults to (8,8).
        cells_per_block (tuple, optional): Cells per block for HOG. Defaults to (2,2).
        visualize (bool, optional): Whether to provide the HOG image. Defaults to False.

    Returns:
        hog_output: array of HOG features, and the HOG image if visualize is True.
    """

    return hog(frame, orientations=orientation, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
               visualize=False, multichannel=True)
