import os
import numpy as np
import skimage.transform
import PIL.Image as pil
from .kittibase import *
from utils.kitti_utils import generate_depth_map, read_calib_file

class KittiDataset(KittiBase):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KittiDataset, self).__init__(*args, **kwargs)

        # NOTE: intrinsics matrix should be *normalized* by the original image size.
        # To normalize, scale the first row by 1 / image_width and the second row
        # by 1 / image_height. A principal point is assumed to be exactly centered.
        # Disable the horizontal flip augmentation if your principal point is far 
        # from the center.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.T = 0.54
        
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def load_calib_info(self, sup_folder, calib_file='calib_cam_to_cam.txt'):
        P_rect_02 = np.eye(4)
        P_rect_03 = np.eye(4)
        calib_info = read_calib_file(os.path.join(self.data_path, sup_folder, calib_file))

        P_rect_02[:-1,:-1] = calib_info['P_rect_02'].reshape(3,4)[:,:-1]
        S_rect_02 = calib_info['S_rect_02']
        P_rect_02[0, :] /= S_rect_02[0]
        P_rect_02[1, :] /= S_rect_02[1]

        P_rect_03[:-1,:-1] = calib_info['P_rect_03'].reshape(3,4)[:,:-1]
        S_rect_03 = calib_info['S_rect_03']
        P_rect_03[0, :] /= S_rect_03[0]
        P_rect_03[1, :] /= S_rect_03[1]

        T_rect_23 = P_rect_02[0,-1]/P_rect_02[0,0] - P_rect_03[0,-1]/P_rect_03[0,0]

        P_rect_02[:2, 2] = 0.5
        P_rect_03[:2, 2] = 0.5

        return {'P_rect_02' : P_rect_02.astype(np.float32),
                                'P_rect_03' : P_rect_03.astype(np.float32),
                                'S_rect_02' : S_rect_02[::-1].astype(np.int32),
                                'S_rect_03' : S_rect_03[::-1].astype(np.int32),
                                'T_rect_23' : T_rect_23}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)
    
    def pil_crop(self, img):
        width, height = img.size
        
        left = (width - self.width) // 2
        right = left + self.width
        top = height - self.height
        bottom = height
        img = img.crop((left, top, right, bottom))
        
        return img

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if self.crop:
            color = self.pil_crop(color)
            
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path
    
    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        # mode=0: nearest-neighbor for downsampling
        # padding mode = 'constant', 'edge'
        h, w = depth_gt.shape

        if self.crop:
            top = h - self.height
            left = (w - self.width) // 2
            depth_gt = depth_gt[top : top + 352, left : left + 1216]
        else:
            depth_gt = skimage.transform.resize(depth_gt, 
                                                self.full_res_shape[::-1], 
                                                order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt     
        
    def get_pred(self, folder, frame_index, do_flip):
        pred_h, pred_w = 352, 1216
        assert do_flip==False, "Guided training is not implemented for flip augmentation for now."
        pred_filename = os.path.join(self.data_path, "pred", folder, f"{frame_index:010d}.bin")
        pre_pred = np.fromfile(pred_filename, dtype=np.float32).reshape(pred_h, pred_w)
        return pre_pred
        
