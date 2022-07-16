from ast import arg
import os
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

import json
from dataset.kittibase import *
from dataset.kittidataset import *
from model import *
from utils.kitti_tool import *
from utils.train_tool import *
from utils.distrib_tool import *
from configs import TrainConfigs
from torch.utils.data import DataLoader, DistributedSampler


"""
The default input size is [B, 3, 352, 1216].
Configure the model accordingly for the input of other shapes
"""


def main(args):
    args.num_scales = len(args.scales)
    init_distributed_mode(args)

    log_folder = os.path.join(args.log_dir, args.model_name)

    # checking height and width are multiples of 32
    assert args.height % 16 == 0, "'height' must be a multiple of 16"
    assert args.width % 16 == 0, "'width' must be a multiple of 16"

    models = {}
    parameters_to_train = []

    device = torch.device(args.device) #default: "cuda"

    #fixed seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    assert args.frame_ids[0] == 0, "frame_ids must start with 0"
    if args.monocular:
        assert not args.stereo, "Stereo and monocular mode should be mutually exclusive."
        use_pose_net = not (args.mono_use_stereo and args.frame_ids == [0]) # no pose net if stereo
        if args.use_stereo:
            args.frame_ids.append("s")

    if args.stereo:
        assert args.frame_ids == [0], "Only one view is needed to find another view."
        args.frame_ids.append("s")

    num_input_frames = len(args.frame_ids)  # for stereo: 2

    model = DepthChiTransformer(
                path=args.weight_path,
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                non_negative=True,
                enable_attention_hooks=False,
                multiscale=True
                )
            
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        try:
            model_without_ddp = model.module
        except AttributeError:
            print("No attribute:module\n")
            model_without_ddp = model.modules

    #set freezed layers:
    if args.freeze_patch_embed:
        for params in model_without_ddp.pretrained.model.vit.patch_embed.parameters():
            params.requires_grad = False

        parameters_to_train = [
            {"params":[params for name, params in model_without_ddp.named_parameters() \
                if "patch_embed" not in name and params.requires_grad]},
        ]
    #...
    else:
        parameters_to_train = [
            {"params":[params for name, params in model_without_ddp.named_parameters() \
                if "patch_embed" not in name and params.requires_grad]},
            {
                "params":[params for name, params in model_without_ddp.named_parameters()\
                    if "patch_embed" in name and params.requires_grad],
                "lr":args.learning_rate_pretrained
            },
        ]
                        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of parameters:', n_parameters)


    optimizer = optim.Adam(parameters_to_train, args.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.scheduler_step_size, 0.1)

    # if args.load_weights_folder is not None:
    #     load_model()

    print("Training model named:\n  ", args.model_name)
    # print("Models and WandB events files are saved to:\n  ", args.log_dir)


    dataset_dict = {"kitti": KittiDataset,
                "cityscapes": "CityScapes"}
    dataset = dataset_dict[args.dataset]

    fpath = os.path.join(os.path.dirname(__file__), "splits", args.split, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    num_train_samples = len(train_filenames)

    img_ext = '.png' if args.png else '.jpg'
    args.num_total_steps = (num_train_samples // (args.batch_size*args.world_size)) * args.num_epochs

    dataset_train = dataset(
        args.data_path, train_filenames, args.height, args.width,
        args.frame_ids, args.num_scales, crop=args.crop, start_scale=0, 
        is_train=True, img_ext=img_ext)
    dataset_val = dataset(
        args.data_path, val_filenames, args.height, args.width,
        args.frame_ids, args.num_scales, crop=args.crop, start_scale=0, 
        is_train=False, img_ext=img_ext)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.batch_sampler:
        sampler_train = torch.util.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        train_loader = DataLoader(
            dataset_train, batch_sampler=sampler_train, #collate_fn
            num_workers=args.num_workers)
    else:
        train_loader = DataLoader(
            dataset_train, args.batch_size, sampler=sampler_train, #shuffle=True,
            num_workers=args.num_workers, drop_last=True)
    
    val_loader = DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val, #shuffle=True,
        num_workers=args.num_workers, drop_last=False)

    if not args.no_ssim:
        ssim = SSIM()
        ssim.to(device)

    backproject_depth = {}
    project_3d = {}
    for scale in args.scales:
        h = args.height // (2 ** scale)
        w = args.width // (2 ** scale)

        backproject_depth[scale] = BackprojectDepth(args.batch_size, h, w)
        backproject_depth[scale].to(device)

        project_3d[scale] = Project3D(args.batch_size, h, w)
        project_3d[scale].to(device)

    depth_metric_names = [
        "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

    print("Using split:\n  ", args.split)
    print("There are {:d} training items and {:d} validation items\n".format(
        len(dataset_train), len(dataset_val)))


    val_iter = iter(val_loader)
    save_opts(args)

    train(args, model, train_loader, val_iter, val_loader, optimizer, 
            lr_scheduler, backproject_depth, project_3d,depth_metric_names, ssim, device)
    
        
def set_train(model):
    if isinstance(model, dict):
        for m in model.values():
            m.train()
    else:
        model.train()

def set_eval(model):
    if isinstance(model, dict):
        for m in model.values():
            m.eval()
    else:
        model.eval()

def train(args, model, train_loader, val_iter, val_loader, optimizer,
         lr_scheduler, backproject_depth, project_3d, depth_metric_names, ssim, device):
    set_train(model)
    print("Start training...")
    args.start_time = time.time()
    args.step = 0
    args.epoch = 0
    for args.epoch in range(args.num_epochs):
        for batch_idx, inputs in enumerate(train_loader):

            before_op_time = time.time()

            for key, values in inputs.items():
                inputs[key] = values.to(device)

            outputs = model(inputs["color_aug", 'l', 0], inputs["color_aug", 'r', 0])
            generate_images_pred(args, inputs, outputs, backproject_depth, project_3d)
            losses = compute_losses(args, inputs, outputs, ssim, device)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % args.log_frequency == 0 and args.step < 2000
            late_phase = args.step % 2000 == 0

            if (early_phase or late_phase) and args.rank==0:
                log_time(args, batch_idx, duration, losses["loss"].cpu().data)
                if "depth_gt" in inputs:
                    compute_depth_losses(args, inputs, outputs, losses, depth_metric_names)

                #log("train", inputs, outputs, losses)
                val(args, model, val_iter, val_loader, backproject_depth, project_3d, depth_metric_names, ssim, device)

            args.step += 1

        lr_scheduler.step()
        if (args.epoch + 1) % args.save_frequency == 0 and args.rank==0:
            save_model(args, model, optimizer)


def process_batch(args, model, inputs, backproject_depth, project_3d, ssim, device):
    """Pass a minibatch through the network and generate images and losses
    """
    for key, ipt in inputs.items():
        inputs[key] = ipt.to(device)

    outputs = model(inputs["color_aug", 'l', 0], inputs["color_aug", 'r', 0])
    generate_images_pred(args, inputs, outputs, backproject_depth, project_3d)
    losses = compute_losses(args, inputs, outputs, ssim, device)

    return outputs, losses

def generate_images_pred(args, inputs, outputs, backproject_depth, project_3d):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    for scale in args.scales:
        depth = outputs[("depth", scale)]
        depth = depth.unsqueeze(1)
        source_scale = 0

        depth = F.interpolate(
                depth, [args.height, args.width], mode="bilinear", align_corners=False)

        outputs[("depth", 0, scale)] = depth

        if args.stereo:
            T = inputs["stereo_T"]
        else:
            T = outputs[("cam_T_cam", 0, 0)]

        cam_points = backproject_depth[source_scale](
            depth, inputs[("inv_K", source_scale)])
        pix_coords = project_3d[source_scale](
            cam_points, inputs[("K", source_scale)], T)

        if args.stereo:
            outputs[("sample", 'r', scale)] = pix_coords

            outputs[("color", 'r', scale)] = F.grid_sample(
                inputs[("color", 'r', source_scale)],
                outputs[("sample", 'r', scale)], align_corners=True,
                padding_mode="border")

        if not args.disable_automasking:
            outputs[("color_identity", 'r', scale)] = \
                inputs[("color", 'r', source_scale)]


def compute_losses(args, inputs, outputs, ssim, device):
    losses = {}
    total_loss = 0

    for scale in args.scales:
        loss = 0
        reprojection_losses = []
        source_scale = 0

        depth = outputs[("depth", scale)].unsqueeze(1)
        color = inputs[("color", 'l', scale)]
        target = inputs[("color", 'l', source_scale)]

        pred = outputs[("color", 'r', scale)]
        reprojection_losses.append(compute_reprojection_loss(args, pred, target, ssim))
        reprojection_losses = torch.cat(reprojection_losses, 1)

        if not args.disable_automasking:
            identity_reprojection_losses = []
        
            pred = inputs[("color", 'r', source_scale)]
            identity_reprojection_losses.append(
                compute_reprojection_loss(args, pred, target, ssim))
        
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if args.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                identity_reprojection_loss = identity_reprojection_losses

        if args.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not args.disable_automasking:
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=device).to(device) * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        if not args.disable_automasking:
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

        loss += to_optimise.mean()

        mean_depth = depth.mean(2, True).mean(3, True)
        norm_depth = depth / (mean_depth + 1e-8)
        smooth_loss = get_smooth_loss(norm_depth, color)

        loss += args.disparity_smoothness * smooth_loss / (2 ** scale)
        total_loss += loss
        losses["loss/{}".format(scale)] = loss

    total_loss /= args.num_scales
    losses["loss"] = total_loss
    return losses


def compute_depth_losses(args, inputs, outputs, losses, depth_metric_names):
    """Compute depth metrics, to allow monitoring during training

    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    depth_pred = outputs[("depth", 0, 0)]
    depth_pred = torch.clamp(
                            F.interpolate(
                                        depth_pred, 
                                        [args.height, args.width], 
                                        mode="bilinear", align_corners=False), 
                            args.min_depth, args.max_depth) 
        #max_depth=80 or 100
    
    depth_pred = depth_pred.detach()

    depth_gt = inputs["depth_gt"]
    mask = depth_gt > 0

    # garg/eigen crop [:, :, 153:371, 44:1197]
    # top: 153-23, btm: 351
    # left: 44-13=31, right:1197-13=1184
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 130:351, 31:1184] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min=args.min_depth, max=args.max_depth)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)

    for i, metric in enumerate(depth_metric_names):
        losses[metric] = np.array(depth_errors[i].cpu())


def compute_reprojection_loss(args, pred, target, ssim):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if args.no_ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


def log_time(args, batch_idx, duration, loss):
    """Print a logging statement to the terminal
    """
    samples_per_sec = args.batch_size / duration
    time_lapse = time.time() - args.start_time
    training_time_left = (args.num_total_steps / args.step - 1.0) * \
        time_lapse if args.step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
        " | loss: {:.5f} | time elapsed: {} | time left: {}"
    print(print_string.format(args.epoch, batch_idx, 
                            samples_per_sec, loss,
                            sec_to_hm_str(time_lapse), sec_to_hm_str(training_time_left)))


def val(args, model, val_iter, val_loader, backproject_depth, project_3d, depth_metric_names, ssim, device):
    """Validate the model on a single minibatch
    """
    set_eval(model)
    try:
        inputs = val_iter.next()
    except StopIteration:
        val_iter = iter(val_loader)
        inputs = val_iter.next()

    with torch.no_grad():
        outputs, losses = process_batch(args, model, inputs, backproject_depth, project_3d, ssim, device)

        if "depth_gt" in inputs:
            compute_depth_losses(args, inputs, outputs, losses, depth_metric_names)

        #log("val", inputs, outputs, losses)
        del inputs, outputs, losses

    set_train(model)


def save_opts(args):
    """Save options to disk so we know what we ran this experiment with
    """
    model_dir = os.path.join(args.log_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    to_save = args.__dict__.copy()

    with open(os.path.join(model_dir, 'opt.json'), 'w') as f:
        json.dump(to_save, f, indent=2)

        
def save_model(args, model, optimizer):
    """Save model weights to disk
    """
    save_folder = os.path.join(args.log_dir, "models", "weights_{}".format(args.epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if isinstance(model, dict):
        for name, mod in model.items():
            save_path = os.path.join(save_folder, "{}.pth".format(name))
            to_save = mod.state_dict()
            if name == 'transformer':
                # save the sizes - these are needed at prediction time
                to_save['height'] = args.height
                to_save['width'] = args.width
                to_save['stereo'] = args.stereo
            torch.save(to_save, save_path)
    else:
        save_path = os.path.join(save_folder, "{}.pth".format(args.model_name))
        torch.save(model.state_dict(), save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(optimizer.state_dict(), save_path)

    
def load_model(args, model, optimizer):
    """Load model(s) from disk
    """
    args.load_weights_folder = os.path.expanduser(args.load_weights_folder)

    assert os.path.isdir(args.load_weights_folder), \
        "Cannot find folder {}".format(args.load_weights_folder)
    print("loading model from folder {}".format(args.load_weights_folder))

    for n in args.models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(args.load_weights_folder, "{}.pth".format(n))
        model_dict = model[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model[n].load_state_dict(model_dict)

    # loading adam state
    optimizer_load_path = os.path.join(args.load_weights_folder, "adam.pth")
    if os.path.isfile(optimizer_load_path):
        print("Loading Adam weights")
        optimizer_dict = torch.load(optimizer_load_path)
        optimizer.load_state_dict(optimizer_dict)
    else:
        print("Cannot find Adam weights so Adam is randomly initialized")



if __name__ == "__main__":
    configs = TrainConfigs()
    args = configs.parse()
    main(args)
