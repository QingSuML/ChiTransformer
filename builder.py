import sys
from functools import partial
from syslog import LOG_SYSLOG

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ChitransformerDepth, ChitransformerDepth_MS
from model import DepthCueRectification_G, DepthCueRectification_Sp

from utils.kitti_utils import get_fp_weight, BackprojectDepth, Project3D, SSIM


class StereoCriterion(nn.Module):
    
    """ This class computes the combination of loss for chitransformer.
    The process happens in x steps:
        1) compute the (multi-scale) reprojection loss w. or w\.
                        edge-aware depth smoothness abd automasking
        2) compute the regularization for learning orthogonal matrices
        3) compute the regularization for learning polarized diagonals
        4) (Optional) compute the masked far-point loss
        5) (Optional) compute the ground truth guided loss
    """
    
    def __init__(self, args, weight_dict, losses, errors=None, embed_dim=768):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            e.g. {"loss" : {"reprojection_loss": 1.0, "fp_loss": 0.3},
                    "reg" : {"orthog_reg": 1e-7, "hoyer_reg": 1e-4}}
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            e.g. ["reprojection_loss", "fp_loss", "orthog_reg", "hoyer_reg"]
            embed_dim: the embedding dimension in attention layers
        """
        super(StereoCriterion, self).__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.errors = errors
        
        #load args
        self.height = args.height
        self.width = args.width
        self.img_scales = args.img_scales
        self.num_dcr = args.num_dcr
        self.dcr_mode = args.dcr_mode
        self.device = args.device
        self.no_ssim = args.no_ssim
        
        if not self.no_ssim:
            self.ssim = SSIM()

        self.grad_ssim = True
            
        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.crop = args.crop

        self.source_scale = args.source_scale
        self.avg_reprojection = args.avg_reprojection

        self.disable_automasking = args.disable_automasking
        self.edge_smoothness= args.edge_smoothness
        
        if self.edge_smoothness:
            self.smoothness_weight = args.smoothness_weight
        
        self.guided_weight = 1.
        
        self.register_buffer("fp_weight", get_fp_weight(self.height, self.width))
        self.register_buffer("ones_vector", torch.ones(embed_dim))
        
        self.backproject_depth = nn.ModuleList([])
        self.project_3d = nn.ModuleList([])
        
        for scale in self.img_scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)
            self.backproject_depth.append(BackprojectDepth(h, w))
            self.project_3d.append(Project3D(h, w))
            
    def forward(self, inputs, outputs, model=None):
        
        self.generate_image_prediction(inputs, outputs)
        
        # Compute all the requested losses
        losses = {}
        
        for loss in self.losses:
            if "loss" in loss:
                losses.update(self.get_loss(loss, inputs, outputs))
            if model and "reg" in loss:
                losses.update(self.get_loss(loss, inputs, outputs, model=model))
                
        return losses

        
    def get_loss(self, loss, inputs, outputs, **kwargs):
        
        loss_map = {
            "reprojection_loss": self.loss_reprojection,
            "fp_loss": self.loss_far_point,
            "guided_loss": self.loss_guided,
            "supervised_loss": self.loss_supervised,
            "orthog_reg": self.orthogonal_regularization,
            "hoyer_reg": self.hoyer_regularization
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        
        if "reg" in loss:
            assert self.dcr_mode in ["spectrum", "sp"], "Regularization is only for spectral decomposition in DCR."
            
        return loss_map[loss](inputs, outputs, **kwargs)
        
        
    def loss_reprojection(self, inputs, outputs):
        losses = 0.

        for scale in self.img_scales:
            
            if self.source_scale:
                depth_scale = outputs[("depth", 0, scale)]
                tgt_scale = inputs[("color", 'l', 0)]
            else:
                depth_scale = outputs[("depth", scale)].unsqueeze(1)
                tgt_scale = inputs[("color", 'l', scale)]
            #if use source_scale, color predictions are in original resolution
            pred_scale = outputs[("color", 'r', scale)]
        
            # reprojection losses [B, 1, h, w]
            reprojection_loss = self.compute_reprojection_loss(pred_scale, tgt_scale)
            
            to_optimise = reprojection_loss
            
            if not self.disable_automasking:
                
                if self.source_scale:
                    tgt_right = inputs[("color", 'r', 0)]
                else:
                    tgt_right = inputs[("color", 'r', scale)]
                    
                left_right_loss = self.compute_reprojection_loss(tgt_right, tgt_scale)
                # inject randomness
                left_right_loss += torch.randn(left_right_loss.shape, device=self.device) * 0.00001
                
                mask = torch.zeros_like(to_optimise, device=self.device)
                mask[to_optimise < left_right_loss] = 1.0
                
                to_optimise = to_optimise * mask * (to_optimise.numel() / mask.sum())
                
            losses += to_optimise.mean() / (2 ** scale)
            
            if self.edge_smoothness:
                # edge-aware depth smoothness
                mean_depth = depth_scale.mean((2,3), keepdim=True)
                norm_depth = depth_scale / (mean_depth + 1e-8)
                smooth_loss = self.compute_smooth_loss(norm_depth, tgt_scale)

                losses += self.smoothness_weight * smooth_loss / (2 ** scale)
                        
            return {"reprojection_loss" : losses}
            

    def loss_far_point(self, inputs, outputs):
    
        mask = self.fp_weighted_mask(inputs[("color", 'l', 0)],
                                inputs[("color", 'r', 0)]) #[B, h, w]
        pred = outputs[("depth", 0)] #[B, h, w]
                
        if mask.sum().item() != 0.:
            fp_to_optimise = pred[mask > 0.]
            #reuse the loaded tensor to save memory, gt is not used here
            fp_value = inputs["depth_gt"].squeeze(1)[mask > 0.] + self.max_depth * 1.5
            fp_loss = F.smooth_l1_loss(fp_to_optimise,  fp_value, reduction="mean")
        else:
            fp_loss = torch.tensor(0., device=self.device).detach()

        return {"fp_loss" : fp_loss}


    def generate_image_prediction(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.img_scales:
            depth = outputs[("depth", scale)]
            depth = depth.unsqueeze(1)

            T = inputs["stereo_T"]

            if self.source_scale:
                #predictions are first upsampled to the original resolution
                depth = F.interpolate(depth, [self.height, self.width], mode="bilinear", align_corners=False)
                outputs[("depth", 0, scale)] = depth

                cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
                outputs[("sample", 'r', scale)] = self.project_3d[0](cam_points, inputs[("K", 0)], T)
                outputs[("color", 'r', scale)] = F.grid_sample(inputs[("color", 'r', 0)],
                                                               outputs[("sample", 'r', scale)],
                                                               align_corners=True,
                                                               padding_mode="border")
                if not self.disable_automasking:
                    outputs[("color_identity", 'r', scale)] = inputs[("color", 'r', 0)]

            else:
                cam_points = self.backproject_depth[scale](depth, inputs[("inv_K", scale)])
                outputs[("sample", 'r', scale)] = self.project_3d[scale](cam_points, inputs[("K", scale)], T)
                outputs[("color", 'r', scale)] = F.grid_sample(inputs[("color", 'r', scale)],
                                                               outputs[("sample", 'r', scale)],
                                                               align_corners=True,
                                                               padding_mode="border")

                if not self.disable_automasking:
                    outputs[("color_identity", 'r', scale)] = inputs[("color", 'r', scale)]
    
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if not self.no_ssim:
            ssim_loss = self.ssim(pred, target).mean(1, True)

            if self.grad_ssim:
                grad_pred_x = F.pad(torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]), 1, keepdim=True), (0,1,0,0), 'replicate')#[352, 1216]
                grad_pred_y = F.pad(torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]), 1, keepdim=True), (0,0,0,1), 'replicate')#[352, 1216]
                grad_pred = grad_pred_x + grad_pred_y

                grad_tgt_x = F.pad(torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True), (0,1,0,0), 'replicate') #[352, 1215]
                grad_tgt_y = F.pad(torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True), (0,0,0,1), 'replicate') #[352, 1216]
                grad_tgt = grad_tgt_x + grad_tgt_y

                ssim_grad_loss = self.ssim(grad_pred, grad_tgt).mean(1, True)

                reprojection_loss = 0.4 * ssim_loss + 0.5 * ssim_grad_loss + 0.1 * l1_loss
            else:
                reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        else:
            reprojection_loss = l1_loss
        
        return reprojection_loss
    
    
    @torch.no_grad()
    def fp_weighted_mask(self, l_img, r_img):
        
        B, _, h, w = l_img.shape

        mask = torch.zeros((B, h, w), device=self.device)
        mask[torch.sum(l_img - r_img, axis=1) == 0.0] = 1.0
        
        for i in range(B):
            mask[i] = -F.max_pool2d(-mask[i][None,...], 9, 1, padding=4)
            mask[i] = F.max_pool2d(mask[i][None,...], 5, 1, padding=2)

        mask = torch.threshold(self.fp_weight*mask, threshold=0.0, value=0.0)

        return mask #[B, h, w]
    
    
    def compute_smooth_loss(self, depth, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:]) #[352, 1215]
        grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :]) #[351, 1216]

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True) #[352, 1215]
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True) #[351, 1216]

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()
        
    def loss_guided(self, inputs, outputs):
        loss = 0.

        pre_pred = inputs[("pre_pred")]
        crop_mask = torch.zeros_like(pre_pred, device=self.device)
        crop_mask[:,:,130:351,31:1184] = 1
        
        for scale in self.img_scales:
            if self.source_scale:
                depth_scale = outputs[("depth", 0, scale)]
            else:
                if scale > 0:
                    raise NotImplementedError
                depth_scale = outputs[("depth", scale)].unsqueeze(1)

            mean_depth = depth_scale.mean((2,3), keepdim=True)
            norm_depth = crop_mask * depth_scale / (mean_depth + 1e-8)

            mean_pred = pre_pred.mean((2,3), keepdim=True)
            pre_pred = crop_mask * pre_pred / (mean_pred + 1e-8)

            smooth_loss = self.compute_smooth_loss(norm_depth, pre_pred)

            loss += self.guided_weight * smooth_loss / (2 ** scale)

        return {"guided_loss" : loss}

    def loss_supervised(self, inputs, outputs):
    
        """Compute guided depth loss at source scale
        """
        depth_gt = inputs["depth_gt"] #[B, 1, h, w]
        
        losses=0.
        
        for scale in self.img_scales:
            if self.source_scale:
                depth_pred = outputs[("depth", 0, scale)] #[B, 1, h, w]
            else:
                depth_pred = outputs[("depth", scale)].unsqueeze(1) #[B, 1, h, w]

            #min_depth=1e-3, max_depth=80.0
            depth_pred = torch.clamp(F.interpolate(depth_pred,
                                                   [self.height, self.width],
                                                   mode="bilinear", align_corners=False),
                                     self.min_depth,)

            mask = (depth_gt > 0) & (depth_gt <= self.max_depth)
            crop_mask = torch.zeros_like(mask, device=self.device)

            if not self.crop:
                # garg/eigen crop
                crop_mask[:, :, 153:371, 44:1197] = 1
            else:
                # top: 153-23, btm: 351
                # left: 44-13=31, right:1197-13=1184
                #top = 153 - (375-self.height) = 130
                #btm = self.height - 1 = 351
                #left = 44 - (1242 - self.width)//2 = 31
                #right = 1197 - (1242 - self.width)//2 = 1184
                crop_mask[:, :, 130:351, 31:1184] = 1
            
            mask = mask * crop_mask

            loss = F.smooth_l1_loss(depth_pred[mask],  depth_gt[mask], reduction="mean")
            
            losses += loss/(2 ** scale)
            
        return {"supervised_loss" : losses}
        
        
    def orthogonal_regularization(self, inputs, outputs, model=None):
        
        orthog_reg = 0.
        
        for module in model.DCR:
            U = module.crs_layer.spectrum.U
            orthog_reg += F.mse_loss(U @ U.T, torch.diag(self.ones_vector), reduction='mean')
        
        orthog_reg /= self.num_dcr
        
        return {"orthog_reg" : orthog_reg}


    #bipolar regularization
    def hoyer_regularization(self,inputs, outputs, model=None):
        
        hoyer_reg = 0.0
        
        for module in model.DCR:
            s1 = module.crs_layer.spectrum.S1
            s2 = module.crs_layer.spectrum.S2
            s_prod = s1 * s2
            reg = F.smooth_l1_loss(s_prod.abs(), self.ones_vector.unsqueeze(-1), reduction='sum')
            hoyer_reg += reg / (s1.norm(2) * s2.norm(2))
            
        hoyer_reg /= self.num_dcr
        
        return {"hoyer_reg" : hoyer_reg}
    
    def compute_depth_errors(self, pred, gt):
        """Computation of error metrics between predicted and ground truth depths
        """
        pred = torch.clamp(pred, max=self.max_depth)
        mask = (gt > 0) & (gt <= self.max_depth)
        crop_mask = torch.zeros_like(mask, device=self.device)

        if not self.crop:
            # garg/eigen crop
            crop_mask[:, 153:371, 44:1197] = 1
        else:
            crop_mask[:, 130:351, 31:1184] = 1

        mask = mask * crop_mask

        gt = gt[mask]
        pred = pred[mask]

        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt) - torch.log(pred + 1e-10)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())

        abs_rel = torch.mean(torch.abs(gt - pred) / gt)

        sq_rel = torch.mean((gt - pred) ** 2 / gt)

        return {"abs_rel" : abs_rel, "sq_rel" : sq_rel, "rmse" : rmse,
                "rmse_log" : rmse_log, "a1" : a1, "a2" : a2, "a3" : a3}


def build(args):

    device = torch.device(args.device)
    
    if args.dcr_mode in ['sp', 'spectral']:
        dcr_mode = partial(DepthCueRectification_Sp, layer_norm=False)
    if args.dcr_mode in ['G', 'direct']:
        dcr_mode = partial(DepthCueRectification_G, layer_norm=False)
    
    model_args = {"in_chans" : args.inchans,
                  "embed_layer" : args.embed_layer,
                  "embed_dim" : args.embed_dim,
                  "depth" : args.depth,
                  "sa_depth" : args.sa_depth,
                  "dcr_module" : dcr_mode,
                  "invert" : args.invert,
                  "scale" : args.scale,
                  "shift" : args.shift,
                  "size" : (args.height, args.width),
                  "device" : device,}
    
    if len(args.img_scales) == 1:
        model = ChitransformerDepth(**model_args)
    else:
        model = ChitransformerDepth_MS(**model_args)

    grid_size = (args.height//16, args.width//16)
    num_patches = grid_size[0]*grid_size[1]
    
    if args.rectilinear_epipolar_geometry:
        for name, values in model.sa_dcr.DCR.named_parameters():
            if "pos_emb" in name:
                values.data = torch.tensor([0.,0.,0.,0.,0.,1.]).unsqueeze(-1).expand(num_patches, -1, -1)
                values.requires_grad_(False) #.to(device)

    if args.dataset == "kitti":
        args.max_depth = 80.0
        args.min_depth = 1e-3
        
        if args.edge_smoothness:
            args.smoothness_weight = 1.0
            
        if args.dcr_mode in ["sp", "spectrum"]:
            weight_dict = {
                "reprojection_loss": 1.5,
                "orthog_reg": 0.1,
                "hoyer_reg": 1e-3,
                "fp_loss" : 5e-5,
                           }
            losses = [
                "reprojection_loss",
                "orthog_reg",
                "hoyer_reg",
                "fp_loss",
                    ]
        else:
            weight_dict = {"reprojection_loss": 1.0, "fp_loss" : 1e-3, "guided_loss":1.0}
            losses = ["reprojection_loss", "fp_loss", "guided_loss"]

        if args.pre_pred > 0:
            weight_dict['guided_loss'] = args.pre_pred ###
            losses.append("guided_loss")

        if args.supervision > 0:
            weight_dict['supervised_loss'] = args.supervision ###
            losses.append("supervised_loss")
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented.")
    
    error_dict = ["abs_rel", "sq_rel", "rmse",  "rmse_log", "a1", "a2", "a3"]
    
    criterion = StereoCriterion(args, weight_dict, losses, error_dict)

    return model, criterion
