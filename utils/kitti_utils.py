import os
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_fp_weight(h, w):
    weight_v = torch.arange(h, 0., -1).unsqueeze(1).expand(-1, w)
    weight_v = 2 * (weight_v / h) - 1.25
    weight_v[weight_v < 0.] = 0.
    weight_h = - (w/2. - torch.arange(w, 0., -1).unsqueeze(0).expand(h, -1)).abs() + w/2.
    weight_h = 2 * (weight_h / (w/2.)) - 1.25
    weight_h[weight_h < 0.] = 0.
    
    weight = weight_v*weight_h
    weight = weight/weight.max()
    
    return weight


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    #velo2cam_0
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., None]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    # Velodynelaser scanner is registered w.r.t the reference camera coordinate system (camera 0)
    R_cam2rect = np.eye(4)
    # rectifying rotation matrix of the reference camera
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    # projection matrix of the cam-th camera
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all points that fall behind the image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera in homogeneous coordinates ((x/z, y/z), z)
    # [N, 3]
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., None]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # bound check
    # e.g. (0.5, 1.5]=>1 => 0, (1241.5, 1242.5] => 1242 => 1241
    # thus, use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1



class BackprojectDepth(nn.Module):
    """Convert a depth image into a point cloud in camera coordinates
    """
    def __init__(self, height, width):
        """ height: height of input image
            width: width of input image
            Not necessary the original resolution (375, 1242)
        """
        
        super(BackprojectDepth, self).__init__()
        
        meshgrid = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        pix_coords = torch.stack((meshgrid[0].contiguous().view(-1), 
                                 meshgrid[1].contiguous().view(-1)), 0)\
                                .unsqueeze(0)\
                                .type(torch.float32)
        #homogeneous coordinates
        pix_coords = torch.cat([pix_coords, torch.ones(1, 1, height*width)], 1)
        
        self.register_buffer('pix_coords', pix_coords) #[1, 3, h*w]
        self.register_buffer('ones', torch.ones((1, 1, 1)))

    def forward(self, depth, inv_K):
                             
        B, _, h, w = depth.shape #[B, 1, h, w]
        #inv_K [B, 4, 4]
                             
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords) #[B, 3, h*w]
        cam_points = depth.view(B, 1, -1) * cam_points
                             
        cam_points = torch.cat([cam_points, self.ones.expand(B, -1, h*w)], 1) # [B, 4, h*w]

        return cam_points


def compute_translation_matrix(translation_vector):
    """Convert a translation vector into a 4-by-4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    vec : [B, 1, 3]
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

    
class Project3D(nn.Module):
    """Projects cloud point into image coords with intrinsics K and translation T 
    """
    def __init__(self, height, width, eps=1e-7):
        """ height: height of input image
            width: width of input image
            Not necessary the original resolution (375, 1242)
        """
        
        super(Project3D, self).__init__()
        
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        #K, T:[B, 4, 4]
        #x:[B, 4, h*w]
        #x' = K.T.x
        B = points.shape[0]
        P = (K @ T)[:, :3, :] #[B, 3, 4]

        cam_points = P @ points #[B, 3, h*w]
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps) #[B, 2, h*w]
        pix_coords = pix_coords.view(B, 2, self.height, self.width) #[B, 2, h, w]
        pix_coords = pix_coords.permute(0, 2, 3, 1) #[B, h, w, 2]
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        
        return pix_coords


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
