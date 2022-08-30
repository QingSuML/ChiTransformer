import torch
import random
from PIL import Image 
import numpy as np
import torch.utils.data as data
from torchvision import transforms



class KittiBase(data.Dataset):
    """Superclass for KITTI dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 start_scale = None,
                 crop = False,
                 is_train=False,
                 load_pred = True, ###
                 img_ext='.jpg'):
        super(KittiBase, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.crop = crop
        self.start_scale = start_scale or 0
        self.num_scales = num_scales
        # Interpolation mode, Image.ANTIALIAS, as a special case of LANCZOS is no longer used
        self.interp = transforms.InterpolationMode.LANCZOS

        self.frame_idxs = frame_idxs #[0, 's'] for stereo
        self.K = None

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.load_pred = load_pred ###
        self.master_side = "l" ###
        self.switch_map = {"r": "l", "l": "r"} ###

        try:
            self.jitter_params = {'brightness': (.8, 1.2), 'contrast': (.8, 1.2), 'saturation': (.8, 1.2),
                                  'hue': (-.1, .1)}
            transforms.ColorJitter.get_params(*self.jitter_params.values())
        except TypeError:
            self.jitter_params = {'brightness': .2, 'contrast': .2, 'saturation': .2, 'hue': .1}

        self.resize = [transforms.Resize((height // 2 ** i, width // 2 ** i),
                                         interpolation=self.interp)
                       for i in range(self.start_scale, self.start_scale + self.num_scales)]
        

        self.load_depth = self.check_depth()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id>:
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width // 2  self.height // 2)
            1       images resized to (self.width // 4, self.height // 4)
            2       images resized to (self.width // 8, self.height // 8)
            3       images resized to (self.width // 16, self.height //16)
        """
        
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
            
        if not self.load_pred: ###
            do_flip = self.is_train and random.random() > 0.5
        else: ###
            do_flip = False ###

        line = self.filenames[index].split()

        assert len(line) == 3, "image keys should composed of 1. diretory, 2.frame_index, 3.side."
        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        for i in self.frame_idxs:
            other_side = self.switch_map[side]
            if i == 's': 
                if do_flip:
                    inputs[( "color", side, self.start_scale - 1)] = self.get_color(folder, frame_index, other_side, do_flip)
                else:
                    inputs[( "color", other_side, self.start_scale - 1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                if do_flip:
                    inputs[("color", other_side, self.start_scale - 1)] = self.get_color(folder, frame_index + i, side, do_flip)
                else:
                    inputs[("color", side, self.start_scale - 1)] = self.get_color(folder, frame_index + i, side, do_flip)


        # adjusting intrinsics to match each scale in the pyramid
        sup_folder = folder.split('/', 1)[0]
        calib_info = self.load_calib_info(sup_folder)
        #img_size = calib_info['S_rect_02']
        if self.K is None:
            K = calib_info['P_rect_02'].copy()
        else:
            K = self.K.copy()
        
        if do_flip:
                K[0,2] = 1 - K[0,2]

        for scale in range(self.start_scale, self.start_scale + self.num_scales):
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(**self.jitter_params)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        
        del inputs[( "color", 'l', self.start_scale - 1)]
        del inputs[( "color_aug", 'l', self.start_scale - 1)]
        del inputs[( "color", 'r', self.start_scale - 1)]
        del inputs[("color_aug", 'r', self.start_scale - 1)]

        if self.load_depth:
            if do_flip: ### if flip, master_side is "r",else "l"
                depth_gt = self.get_depth(folder, frame_index, self.switch_map[self.master_side], do_flip) ###
            else: ###
                depth_gt = self.get_depth(folder, frame_index, self.master_side, do_flip) ###
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if self.load_pred > 0:
            pre_pred = self.get_pred(folder, frame_index, do_flip) #[352, 1216]
            pre_pred = torch.from_numpy(pre_pred[None, ...])
            inputs["pre_pred"] = pre_pred

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            stereo_T[0, 3] = - (self.T or -0.54)
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs
    
    def preprocess(self, inputs, color_aug):
        for key in list(inputs):
            if "color" in key:
                c, s, _ = key
                for i in range(self.start_scale, self.start_scale + self.num_scales):
                    inputs[(c, s, i)] = self.resize[i - self.start_scale](inputs[(c, s, i - 1)])

        for key in list(inputs):
            if "color" in key:
                c, s, i = key
                frame = inputs[key]
                inputs[key] = self.to_tensor(frame)
                inputs[(c + "_aug", s, i)] = self.to_tensor(color_aug(frame))
                
    
    def pil_crop(self, img):
        raise NotImplementedError
    
    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_pred(self, folder, frame_index, do_flip):
        raise NotImplementedError
    
    def load_calib_info(self, sup_folder, calib_file='calib_cam_to_cam.txt'):
        raise NotImplementedError
        
        
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines